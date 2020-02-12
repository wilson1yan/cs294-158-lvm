from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from deepul_helper.data import sample_four_blobs
from deepul_helper.visualize import plot_scatter_2d
from deepul_helper.models import MLP
from deepul_helper.trainer import train_epochs

train_data = sample_four_blobs(1000)
test_data = sample_four_blobs(250)
plot_scatter_2d(train_data, title='Train Data')

class SimpleLVM(nn.Module):
    def __init__(self, n_mix):
        super().__init__()

        self.n_mix = n_mix
        self.pi_logits = nn.Parameter(torch.zeros(n_mix, dtype=torch.float32), requires_grad=True)
        self.net = MLP(n_mix, 2, hiddens=[64, 64])

        self.one_hot = torch.eye(n_mix, dtype=torch.float32).to(device)

    def loss(self, x):
        mu = self.net(self.one_hot)
        log_probs = []
        for i in range(self.n_mix):
            mu_i = mu[i].unsqueeze(0)
            log_prob = (-0.5 * (x - mu_i) ** 2).sum(1)
            log_prob = log_prob - 0.5 * np.log(2 * np.pi)
            log_probs.append(log_prob)
        log_probs = torch.stack(log_probs, dim=1)

        log_pi = F.log_softmax(self.pi_logits, dim=0)
        log_probs = log_probs + log_pi.unsqueeze(0)
        loss = -torch.logsumexp(log_probs, dim=1).mean()
        return OrderedDict(loss=loss)

    def sample(self, n):
        with torch.no_grad():
            probs = F.softmax(self.pi_logits, dim=0)
            labels = torch.multinomial(probs, n, replacement=True)
            one_hots = torch.zeros(n, self.n_mix, dtype=torch.float32)
            one_hots.scatter_(1, labels.unsqueeze(1), 1)
            mu = self.net(one_hots)
            x = torch.randn(n, 2) + mu
        return x.numpy(), labels.numpy()

train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = data.DataLoader(test_data, batch_size=128)
device = torch.device('cpu')

n_mix = 4
model = SimpleLVM(n_mix)

def fn(epoch):
    x, labels = model.sample(10000)
    plot_scatter_2d(x, title=f'Epoch {epoch} Samples', labels=labels)

train_epochs(model, train_loader, test_loader, device, dict(epochs=10, lr=2e-3),
             fn=fn, fn_every=2)

x, labels = model.sample(10000)
plot_scatter_2d(x, title='Final Samples', labels=labels)


