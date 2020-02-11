from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
                 dec_dist, prior, enc_hidden_sizes=[], dec_hidden_sizes=[], beta=1):
        super().__init__()
        if isinstance(input_shape, int):
            input_shape = (input_shape,)

        self.input_shape = input_shape
        self.latent_size = latent_size
        self.beta = beta

        self.dec_dist = dec_dist
        self.enc_dist = enc_dist
        self.prior = prior

        self.encoder = MLP(input_shape, get_dist_output_size(enc_dist, latent_size, flattened=True),
                           enc_hidden_sizes)
        self.decoder = MLP(latent_size, get_dist_output_size(dec_dist, input_shape, flattened=True),
                           dec_hidden_sizes)


class ConvEncoder(nn.Module):
    def __init__(self, input_shape, output_size, conv_sizes=[(3, 64, 2)]):
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
        self.fc = nn.Linear(prev_filters * int(out_size) ** 2, output_size)

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.shape[0], -1)
        return self.fc(out)


class ConvDecoder(nn.Module):
    def __init__(self, input_size, output_channels, base_size, conv_sizes=[(4, 64, 2, 2)]):
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
        self.convs.append(nn.Conv2d(prev_filters, output_channels, 3, padding=1))
        self.convs = nn.Sequential(*self.convs)

    def forward(self, z):
        out = F.relu(self.fc(z)).view(-1, *self.base_size)
        out = self.convs(out)
        return out


class ConvVAE(VAE):
    def __init__(self, input_shape, latent_size, enc_dist,
                 dec_dist, prior, enc_conv_sizes=[(3, 64, 2)], dec_base_size=(16, 7, 7),
                 dec_conv_sizes=[(4, 64, 2, 1)], beta=1):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
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


# AF-VAE

# IWAE

# VQ-VAE


# Gumbel-Softmax VAE