from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from deepul_helper.data import sample_three_blobs
from deepul_helper.visualize import plot_scatter_2d
from deepul_helper.models import MLP
from deepul_helper.trainer import train_epochs

train_data = sample_three_blobs(10000)
test_data = sample_three_blobs(2500)
# plot_scatter_2d(train_data, title='Train Data')

class SimpleLVM(nn.Module):
    def __init__(self, n_mix):
        super().__init__()

        self.n_mix = n_mix
        self.pi_logits = nn.Parameter(torch.zeros(n_mix, dtype=torch.float32), requires_grad=True)
        self.mus = nn.Parameter(torch.randn(n_mix, 2, dtype=torch.float32), requires_grad=True)
        self.log_stds = nn.Parameter(-torch.ones(n_mix, 2, dtype=torch.float32), requires_grad=True)

    def loss(self, x):
        log_probs = []
        for i in range(self.n_mix):
            mu_i, log_std_i = self.mus[i].unsqueeze(0), self.log_stds[i].unsqueeze(0)
            log_prob = -0.5 * (x - mu_i) ** 2 * torch.exp(-2 * log_std_i)
            log_prob = log_prob - 0.5 * np.log(2 * np.pi) - log_std_i
            log_probs.append(log_prob.sum(1))
        log_probs = torch.stack(log_probs, dim=1)

        log_pi = F.log_softmax(self.pi_logits, dim=0)
        log_probs = log_probs + log_pi.unsqueeze(0)
        loss = -torch.logsumexp(log_probs, dim=1).mean()
        return OrderedDict(loss=loss)

    def sample(self, n):
        with torch.no_grad():
            probs = F.softmax(self.pi_logits, dim=0)
            labels = torch.multinomial(probs, n, replacement=True)
            mus, log_stds = self.mus[labels], self.log_stds[labels]
            x = torch.randn(n, 2) * log_stds.exp() + mus
        return x.numpy(), labels.numpy()

train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = data.DataLoader(test_data, batch_size=128)
device = torch.device('cpu')

n_mix = 3
model = SimpleLVM(n_mix)

def fn(epoch):
    x, labels = model.sample(10000)
    plot_scatter_2d(x, title=f'Epoch {epoch} Samples', labels=labels)

train_epochs(model, train_loader, test_loader, device, dict(epochs=10, lr=5e-2),
             fn=fn, fn_every=2)

x, labels = model.sample(10000)
plot_scatter_2d(x, title='Final Samples', labels=labels)


