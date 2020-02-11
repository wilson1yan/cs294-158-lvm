from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepul_helper.distributions import kl, get_dist_output_size


class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, hiddens=[]):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hiddens = hiddens

        model = []
        prev_h = np.prod(input_shape)
        for h in hiddens + [np.prod(output_shape)]:
            model.append(nn.Linear(prev_h, h))
            model.append(nn.LeakyReLU(0.2))
            model.append(nn.Dropout(0.1))
            prev_h = h
        model.pop()
        self.net = nn.Sequential(*model)

    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, -1)
        return self.net(x).view(b, *self.output_shape)


# Generic VAE implementation
class VAE(nn.Module):
    def encode(self, x, sample=True):
        out = self.encoder(x)
        if sample:
            z = self.enc_dist.sample(out)
        else:
            z = self.enc_dist.expectation(out)
        return z

    def decode(self, z, sample=True):
        out = self.decoder(z)
        if sample:
            x = self.dec_dist.sample(out)
        else:
            x = self.dec_dist.expectation(out)
        return x.view(-1, *self.input_shape)

    def forward(self, x):
        enc_params = self.encoder(x)
        z = self.enc_dist.sample(enc_params)
        dec_params = self.decoder(z)
        dec_params = self.output_activation(dec_params)
        return z, enc_params, dec_params

    def loss(self, x):
        z, enc_params, dec_params = self(x)

        self.enc_dist.set_params(enc_params)
        self.dec_dist.set_params(dec_params)

        recon_loss = -self.dec_dist.log_prob(x)
        kl_loss = kl(z, self.enc_dist, self.prior)
        recon_loss, kl_loss = recon_loss.mean(), kl_loss.mean()

        return OrderedDict(loss=recon_loss + self.beta * kl_loss, recon_loss=recon_loss,
                           kl_loss=kl_loss)

    def sample(self, n, decoder_noise=True):
        with torch.no_grad():
            z = torch.cat([self.prior.sample() for _ in range(n)], dim=0)
            return self.decode(z, sample=decoder_noise).cpu()



class FullyConnectedVAE(VAE):
    def __init__(self, input_shape, latent_size, enc_dist,
                 dec_dist, prior, enc_hidden_sizes=[], dec_hidden_sizes=[],
                 output_activation=lambda x: x, beta=1):
        super().__init__()
        if isinstance(input_shape, int):
            input_shape = (input_shape,)

        self.input_shape = input_shape
        self.latent_size = latent_size
        self.output_activation = output_activation
        self.beta = beta

        self.dec_dist = dec_dist
        self.enc_dist = enc_dist
        self.prior = prior

        self.encoder = MLP(input_shape, get_dist_output_size(enc_dist, latent_size),
                           enc_hidden_sizes)
        self.decoder = MLP(latent_size, get_dist_output_size(dec_dist, input_shape),
                           dec_hidden_sizes)


class ConvEncoder(nn.Module):
    def __init__(self, input_shape, output_shape, conv_sizes=[(3, 64, 2)]):
        super().__init__()
        assert input_shape[1] == input_shape[2]

        self.convs = []
        out_size = input_shape[1]
        prev_filters = input_shape[0]
        for k, filters, stride in conv_sizes:
            assert k % 2 == 1
            self.convs.append(nn.Conv2d(prev_filters, filters, k, padding=k // 2, stride=stride))
            self.convs.append(nn.LeakyReLU(0.2))
            self.convs.append(nn.Dropout2d(0.5))
            prev_filters = filters
            out_size /= stride
        self.convs = nn.Sequential(*self.convs)
        self.fc = nn.Linear(prev_filters * int(out_size) ** 2, output_shape[0])

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.shape[0], -1)
        return self.fc(out)


class ConvDecoder(nn.Module):
    def __init__(self, input_size, output_shape, base_size, conv_sizes=[(4, 64, 2, 2)]):
        super().__init__()

        self.base_size = base_size
        self.fc = nn.Linear(input_size, np.prod(base_size))
        self.convs = []
        prev_filters = base_size[0]
        for k, filters, stride, padding in conv_sizes:
            self.convs.append(nn.ConvTranspose2d(prev_filters, filters, k, stride=stride, padding=padding))
            self.convs.append(nn.LeakyReLU(0.2))
            self.convs.append(nn.Dropout2d(0.5))
            prev_filters = filters
        self.convs.append(nn.Conv2d(prev_filters, output_shape[0], 3, padding=1))
        self.convs = nn.Sequential(*self.convs)

    def forward(self, z):
        out = F.relu(self.fc(z)).view(-1, *self.base_size)
        out = self.convs(out)
        return out


class ConvVAE(VAE):
    def __init__(self, input_shape, latent_size, enc_dist,
                 dec_dist, prior, enc_conv_sizes=[(3, 64, 2)], dec_base_size=(16, 7, 7),
                 dec_conv_sizes=[(4, 64, 2, 1)], output_activation=lambda x: x, beta=1):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        self.output_activation = output_activation
        self.beta = beta

        self.dec_dist = dec_dist
        self.enc_dist = enc_dist
        self.prior = prior

        self.encoder = ConvEncoder(input_shape, get_dist_output_size(enc_dist, latent_size),
                                   enc_conv_sizes)
        self.decoder = ConvDecoder(latent_size, get_dist_output_size(dec_dist, input_shape),
                                   dec_base_size, dec_conv_sizes)


# MADE
# Code based one Andrej Karpathy's implementation: https://github.com/karpathy/pytorch-made
class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True, conditional_size=None):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

        if conditional_size is not None:
            self.cond_op = nn.Linear(conditional_size, out_features)

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input, cond=None):
        out = F.linear(input, self.mask * self.weight, self.bias)
        if cond is not None:
            out = out + self.cond_op(input)
        return out

class MADE(nn.Module):
  def __init__(self, input_shape, d, hidden_size=[32, 32], ordering=None,
               conditional_size=None):
    super().__init__()
    self.input_shape = input_shape
    self.nin = np.prod(input_shape)
    self.nout = self.nin * d
    self.d = d
    self.hidden_sizes = hidden_size
    self.ordering = np.arange(self.nin) if ordering is None else ordering

    # define a simple MLP neural net
    self.net = []
    hs = [self.nin] + self.hidden_sizes + [self.nout]
    for h0, h1 in zip(hs, hs[1:]):
      self.net.extend([
        MaskedLinear(h0, h1, conditional_size=conditional_size),
        nn.ReLU(),
      ])
    self.net.pop()  # pop the last ReLU for the output layer
    self.net = nn.ModuleList(self.net)

    self.m = {}
    self.create_mask()  # builds the initial self.m connectivity

  def create_mask(self):
    L = len(self.hidden_sizes)

    # sample the order of the inputs and the connectivity of all neurons
    self.m[-1] = self.ordering
    for l in range(L):
      self.m[l] = np.random.randint(self.m[l - 1].min(),
                                      self.nin - 1, size=self.hidden_sizes[l])

    # construct the mask matrices
    masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
    masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

    masks[-1] = np.repeat(masks[-1], self.d, axis=1)

    # set the masks in all MaskedLinear layers
    layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
    for l, m in zip(layers, masks):
      l.set_mask(m)

  def forward(self, x, cond=None):
    batch_size = x.shape[0]
    out = x.view(batch_size, self.nin)
    for layer in self.net:
        if isinstance(out, MaskedLinear):
            out = layer(out, cond=cond)
        else:
            out = layer(out)
    out = out.view(batch_size, self.nin, self.d)
    return out


class IAFEncoder(nn.Module):
    def __init__(self, device, input_shape, embedding_size, latent_size,
                 base_hidden_sizes=[], made_hidden_sizes=[128]):
        super().__init__()
        self.input_shape = input_shape
        self.embedding_size = embedding_size
        self.latent_size = latent_size
        self.device = device

        self.base_dist = MLP(input_shape, embedding_size + 2 * latent_size,
                             base_hidden_sizes)
        self.made = MADE(latent_size, 2 * latent_size, made_hidden_sizes,
                         conditional_size=embedding_size)

    def forward(self, x):
        eps = torch.randn(x.shape[0], self.latent_size)
        out = self.base_dist(x)
        h = out[:, :self.embedding_size]
        base_mu, base_log_stddev = out[:, self.embedding_size:].chunk(2, dim=1)
        z = eps * base_log_stddev.exp() + base_mu
        log_prob = 0.5 * np.log(2 * np.pi) + base_log_stddev + 0.5 * eps ** 2

        for i in range(self.latent_size):
            out = self.made(z, cond=h)[:, i]
            mu, s = out.chunk(2, dim=1)
            s = F.logsigmoid(s)
            z = s.exp() * z + (1 - s.exp()) * mu
            log_prob = log_prob + s
        return z, -log_prob.sum(dim=1)


# IAF-VAE
class FullyConnectedIAFVAE(nn.Module):
    def __init__(self, device, input_shape, embedding_size, latent_size, dec_dist,
                 base_hidden_sizes=[], made_hidden_sizes=[], dec_hidden_sizes=[], beta=1):
        super().__init__()
        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        self.input_shape = input_shape
        self.latent_size = latent_size
        self.embedding_size = embedding_size
        self.device = device
        self.beta = beta

        self.dec_dist = dec_dist

        self.encoder = IAFEncoder(device, input_shape, embedding_size, latent_size,
                                  base_hidden_sizes, made_hidden_sizes)
        self.decoder = MLP(latent_size, get_dist_output_size(dec_dist, input_shape),
                           dec_hidden_sizes)

    def encode(self, x, sample=True): # sample does nothing, just there to keep input consistent
        z, _ = self.encoder(x)
        return z

    def decode(self, z, sample=True):
        out = self.decoder(z)
        if sample:
            x = self.dec_dist.sample(out)
        else:
            x = self.dec_dist.expectation(out)
        return x.view(-1, *self.input_shape)

    def forward(self, x):
        z, z_log_prob = self.encoder(x)
        dec_params = self.decoder(z)
        return z, z_log_prob, dec_params

    def loss(self, x):
        z, z_log_prob, dec_params = self(x)
        self.dec_dist.set_params(dec_params)
        recon_loss = -self.dec_dist.log_prob(x)
        prior_log_prob = -(0.5 * np.log(2 * np.pi) + 0.5 * z ** 2).sum(dim=1)
        kl_loss = z_log_prob - prior_log_prob
        recon_loss, kl_loss = recon_loss.mean(), kl_loss.mean()

        return OrderedDict(loss=recon_loss + self.beta * kl_loss, recon_loss=recon_loss,
                           kl_loss=kl_loss)

# Sentence VAE - code from https://github.com/timbmg/Sentence-VAE
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class SentenceVAE(nn.Module):

    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1, bidirectional=False):

        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

    def forward(self, batch):
        input_sequence, length = batch['input'], batch['length']

        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        # ENCODER
        input_embedding = self.embedding(input_sequence)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = torch.randn([batch_size, self.latent_size])
        z = z * std + mean

        # DECODER
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder input
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size())
            if torch.cuda.is_available():
                prob=prob.cuda()
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence)
        input_embedding = self.embedding_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b,s,_ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)


        return logp, mean, logv, z


    def loss(self, batch):
        target = batch['target']

        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).data[0]].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))

        # Negative Log Likelihood
        NLL_loss = NLL(logp, target)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_anneal_function(anneal_function, step, k, x0)

        return NLL_loss, KL_loss, KL_weight


    def inference(self, n=4, z=None):

        if z is None:
            batch_size = n
            z = torch.randn([batch_size, self.latent_size])
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size).long() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size).byte()

        running_seqs = torch.arange(0, batch_size).long().cuda() # idx of still generating sequences with respect to current loop

        generations = torch.zeros(batch_size, self.max_sequence_length).long().cuda()

        t=0
        while(t<self.max_sequence_length and len(running_seqs)>0):

            if t == 0:
                input_sequence = torch.Tensor(batch_size).fill_(self.sos_idx).long()

            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs)).long().cuda()

            t += 1

        return generations, z

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

# AF-VAE

# IWAE


# PixelCNN
class MaskConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        assert mask_type == 'A' or mask_type == 'B'
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask(mask_type)

    def forward(self, input, cond=None):
        out = F.conv2d(input, self.weight * self.mask, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        return out

    def create_mask(self, mask_type):
        k = self.kernel_size[0]
        self.mask[:, :, :k // 2] = 1
        self.mask[:, :, k // 2, :k // 2] = 1
        if mask_type == 'B':
            self.mask[:, :, k // 2, k // 2] = 1


class PixelCNN(nn.Module):
    def __init__(self, device, d, input_shape=(1, 7, 7), kernel_size=5, n_layers=7):
        super().__init__()
        assert n_layers >= 2

        model = nn.ModuleList([MaskConv2d('A', input_shape[0], 64, kernel_size,
                                          padding=kernel_size // 2), nn.ReLU()])
        for _ in range(n_layers - 2):
            model.extend([MaskConv2d('B', 64, 64, kernel_size,
                                     padding=kernel_size // 2), nn.ReLU()])
        model.append(MaskConv2d('B', 64, input_shape[0] * d, kernel_size, padding=kernel_size // 2))
        self.net = model
        self.d = d
        self.device = device
        self.input_shape = input_shape

    def forward(self, x):
        out = 2 * (x.float() / (self.d - 1)) - 1
        for layer in self.net:
            out = layer(out)
        return out.view(x.shape[0], self.d, *self.input_shape)

    def loss(self, x):
        return OrderedDict(loss=F.cross_entropy(self(x), x.long()))

    def sample(self, n):
        samples = torch.zeros(n, *self.input_shape).to(self.device)
        with torch.no_grad():
            for r in range(self.input_shape[1]):
                for c in range(self.input_shape[2]):
                    for k in range(self.input_shape[0]):
                        logits = self(samples)[:, :, k, r, c]
                        logits = F.softmax(logits, dim=1)
                        samples[:, k, r, c] = torch.multinomial(logits, 1).squeeze(-1)
        return samples


# VQ-VAE
class Quantize(nn.Module):

    def __init__(self, size, code_dim, gamma=0.99):
        super().__init__()
        self.embedding = nn.Embedding(size, code_dim)
        self.embedding.weight.requires_grad = False

        self.code_dim = code_dim
        self.size = size
        self.gamma = gamma

        self.register_buffer('N', torch.zeros(size))
        self.register_buffer('z_avg', self.embedding.weight.data.clone())

    def forward(self, z):
        b, c, h, w = z.shape
        weight = self.embedding.weight

        flat_inputs = z.permute(0, 2, 3, 1).contiguous().view(-1, self.code_dim)
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                    - 2 * torch.mm(flat_inputs, weight.t()) \
                    + (weight.t() ** 2).sum(dim=0, keepdim=True)
        encoding_indices = torch.max(-distances, dim=1)[1]
        encode_onehot = F.one_hot(encoding_indices, self.size).type(flat_inputs.dtype)
        encoding_indices = encoding_indices.view(b, h, w)
        quantized = self.embedding(encoding_indices).permute(0, 3, 1, 2)

        if self.training:
            self.N.data.mul_(self.gamma).add_(1 - self.gamma, encode_onehot.sum(0))

            encode_sum = torch.mm(flat_inputs.t(), encode_onehot)
            self.z_avg.data.mul_(self.gamma).add_(1 - self.gamma, encode_sum.t())

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.size * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embedding.weight.data.copy_(encode_normalized)

        return quantized, (quantized - z).detach() + z, encoding_indices

class VectorQuantizedVAE(nn.Module):
    def __init__(self, code_dim, code_size, beta):
        super().__init__()
        self.beta = beta
        self.code_size = code_size

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, code_dim, 5, stride=2, padding=2)
        )

        self.codebook = Quantize(code_size, code_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(code_dim, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Tanh(),
        )

    def encode_code(self, x):
        z = self.encoder(x)
        indices = self.codebook(z)[2]
        return indices

    def decode_code(self, latents):
        latents = self.codebook.embedding(latents).permute(0, 3, 1, 2)
        return self.decoder(latents)

    def forward(self, x):
        z = self.encoder(x)
        e, e_st, _ = self.codebook(z)
        x_tilde = self.decoder(e_st)

        diff = (z - e.detach()).pow(2).mean()
        return x_tilde, diff

    def loss(self, x):
        x_tilde, diff = self(x)
        recon_loss = F.mse_loss(x_tilde, x)
        diff_loss = diff
        loss = recon_loss + self.beta * diff_loss
        return OrderedDict(loss=loss, recon_loss=recon_loss, diff_loss=diff_loss)

# Gumbel-Softmax VAE