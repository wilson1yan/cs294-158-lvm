from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from deepul_helper.distributions import normal_kl, normal_log_prob


class MLP(nn.Module):
    def __init__(self, input_shape, output_size, hiddens=[]):
        super().__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.hiddens = hiddens

        model = []
        prev_h = np.prod(input_shape)
        for h in hiddens + [output_size]:
            model.append(nn.Linear(prev_h, h))
            model.append(nn.LeakyReLU(0.2))
            prev_h = h
        model.pop()
        self.net = nn.Sequential(*model)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.net(x)


class MixtureVAE(nn.Module):
    def __init__(self, device, input_shape, latent_size,
                 enc_hidden_sizes=[], dec_hidden_sizes=[], beta=1):
        super().__init__()
        if isinstance(input_shape, int):
            input_shape = (input_shape,)

        self.input_shape = input_shape
        self.latent_size = latent_size
        self.device = device
        self.beta = beta

        self.encoder = MLP(input_shape, 2 * latent_size, enc_hidden_sizes)
        self.decoder = MLP(latent_size, 2 * np.prod(input_shape), dec_hidden_sizes)

    def encode(self, x, sample=True):
        mu, log_stddev = self.encoder(x).chunk(2, dim=1)
        if sample:
            return torch.randn_like(mu) * log_stddev.exp() + mu
        return mu

    def decode(self, z, sample=True):
        out = self.decoder(z)
        mu, log_stddev = out.chunk(2, dim=1)

        if sample:
            eps = torch.randn_like(mu)
            x = mu + eps * log_stddev.exp()
        else:
            x = mu
        return x.view(-1, *self.input_shape)

    def forward(self, x):
        out = self.encoder(x)
        z_mu, z_log_stddev = out.chunk(2, dim=1)
        z_log_stddev = torch.tanh(z_log_stddev)
        eps = torch.randn_like(z_mu)
        z = z_mu + eps * z_log_stddev.exp()

        out = self.decoder(z)
        x_recon_mu, x_recon_log_stddev = out.chunk(2, dim=1)
        x_recon_log_stddev = torch.tanh(x_recon_log_stddev)
        x_recon_mu = x_recon_mu.view(-1, *self.input_shape)
        x_recon_log_stddev = x_recon_log_stddev.view(-1, *self.input_shape)

        return z_mu, z_log_stddev, x_recon_mu, x_recon_log_stddev

    def loss(self, x):
        z_mu, z_log_stddev, x_recon_mu, x_recon_log_stddev = self(x)

        recon_loss = -normal_log_prob(x, x_recon_mu, x_recon_log_stddev)
        kl_loss = normal_kl(z_mu, z_log_stddev, torch.zeros_like(z_mu), torch.ones_like(z_log_stddev))
        recon_loss, kl_loss = recon_loss.mean(), kl_loss.mean()

        return OrderedDict(loss=recon_loss + self.beta * kl_loss, recon_loss=recon_loss,
                           kl_loss=kl_loss)

    def sample(self, n):
        with torch.no_grad():
            z = torch.randn(n, self.latent_size).to(self.device)
            return self.decode(z).cpu()
