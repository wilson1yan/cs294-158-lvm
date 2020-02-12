import json
import torch
import numpy as np
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from deepul_helper.text_vae.ptb import PTB
from deepul_helper.text_vae.utils import to_var, idx2word, interpolate
from deepul_helper.text_vae.model import SentenceVAE


splits = ['train', 'valid', 'test']

datasets = OrderedDict()
for split in splits:
    datasets[split] = PTB(
        data_dir='data',
        split=split,
        create_data=True,
        max_sequence_length=60,
        min_occ=1
    )

model = SentenceVAE(
    vocab_size=datasets['train'].vocab_size,
    sos_idx=datasets['train'].sos_idx,
    eos_idx=datasets['train'].eos_idx,
    pad_idx=datasets['train'].pad_idx,
    unk_idx=datasets['train'].unk_idx,
    max_sequence_length=60,
    embedding_size=300,
    rnn_type='gru',
    hidden_size=256,
    word_dropout=0,
    embedding_dropout=0.5,
    latent_size=16,
    num_layers=1,
    bidirectional=False
)

if torch.cuda.is_available():
    model = model.cuda()

def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == 'linear':
        return min(1, step / x0)

NLL = torch.nn.NLLLoss(size_average=False, ignore_index=datasets['train'].pad_idx)

def loss_fn(logp, target, length, mean, logv, anneal_function, step, k, x0):

    # cut-off unnecessary padding from target, and flatten
    target = target[:, :torch.max(length).item()].contiguous().view(-1)
    logp = logp.view(-1, logp.size(2))

    # Negative Log Likelihood
    NLL_loss = NLL(logp, target)

    # KL Divergence
    KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    KL_weight = kl_anneal_function(anneal_function, step, k, x0)

    return NLL_loss, KL_loss, KL_weight

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
step = 0
with open('data/ptb.vocab.json', 'r') as file:
    vocab = json.load(file)

w2i, i2w = vocab['w2i'], vocab['i2w']
for epoch in range(10):
    model.train()
    for split in splits:

        data_loader = DataLoader(
            dataset=datasets[split],
            batch_size=32,
            shuffle=split == 'train',
            num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )

        tracker = defaultdict(tensor)

        # Enable/Disable Dropout
        if split == 'train':
            model.train()
        else:
            model.eval()

        for iteration, batch in enumerate(data_loader):

            batch_size = batch['input'].size(0)

            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = to_var(v)

            # Forward pass
            logp, mean, logv, z = model(batch['input'], batch['length'])

            # loss calculation
            NLL_loss, KL_loss, KL_weight = loss_fn(logp, batch['target'],
                                                   batch['length'], mean, logv, 'logistic', step, 0.0025,
                                                   2500)

            loss = (NLL_loss + KL_weight * KL_loss) / batch_size

            # backward + optimization
            if split == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1

            # if iteration % args.print_every == 0 or iteration + 1 == len(data_loader):
            #     print("%s Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
            #           % (
            #           split.upper(), iteration, len(data_loader) - 1, loss.item(), NLL_loss.item() / batch_size,
            #           KL_loss.item() / batch_size, KL_weight))
    print(f'Completed Epoch {epoch}')

    model.eval()

    samples, z = model.inference(n=10)
    print('----------SAMPLES----------')
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

    z1 = torch.randn([16]).numpy()
    z2 = torch.randn([16]).numpy()
    z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())
    samples, _ = model.inference(z=z)
    print('-------INTERPOLATION-------')
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
