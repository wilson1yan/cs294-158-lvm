from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from deepul_helper.distributions import kl, get_dist_output_size


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


class FullyConnectedVAE(nn.Module):
    def __init__(self, device, input_shape, latent_size, enc_dist,
                 dec_dist, prior, enc_hidden_sizes=[], dec_hidden_sizes=[], beta=1):
        super().__init__()
        if isinstance(input_shape, int):
            input_shape = (input_shape,)

        self.input_shape = input_shape
        self.latent_size = latent_size
        self.device = device
        self.beta = beta

        self.dec_dist = dec_dist
        self.enc_dist = enc_dist
        self.prior = prior

        self.encoder = MLP(input_shape, get_dist_output_size(enc_dist, latent_size),
                           enc_hidden_sizes)
        self.decoder = MLP(latent_size, get_dist_output_size(dec_dist, input_shape),
                           dec_hidden_sizes)

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
