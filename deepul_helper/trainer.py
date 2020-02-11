from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim


def train(model, train_loader, optimizer, epoch, device, grad_clip=None):
    model.train()

    pbar = tqdm(total=len(train_loader.dataset))
    losses = OrderedDict()
    for x in train_loader:
        if isinstance(x, list):
            x = x[0]
        x = x.to(device)
        out = model.loss(x)
        optimizer.zero_grad()
        out['loss'].backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        desc = f'Epoch {epoch}'
        for k, v in out.items():
            if k not in losses:
                losses[k] = []
            losses[k] += [v.item()]
            avg_loss = np.mean(losses[k][-50:])
            desc += f', {k} {avg_loss:.4f}'

        pbar.set_description(desc)
        pbar.update(x.shape[0])
    pbar.close()


def eval_loss(model, data_loader, device):
    model.eval()
    total_losses = OrderedDict()
    with torch.no_grad():
        for x in data_loader:
            if isinstance(x, list):
                x = x[0]
            x = x.to(device)
            out = model.loss(x)
            for k, v in out.items():
                total_losses[k] = total_losses.get(k, 0) + v.item() * x.shape[0]

        desc = 'Test '
        for k in total_losses.keys():
            total_losses[k] /= len(data_loader.dataset)
            desc += f', {k} {total_losses[k]:.4f}'
        print(desc)


def train_epochs(model, train_loader, test_loader, device, train_args):
    epochs, lr = train_args['epochs'], train_args['lr']
    grad_clip = train_args.get('grad_clip', None)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train(model, train_loader, optimizer, epoch, device, grad_clip)
        if test_loader is not None:
            eval_loss(model, test_loader, device)
