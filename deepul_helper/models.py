from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepul_helper.distributions import kl, get_dist_output_size


class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, hiddens=[]):
        super().__init__()

        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        if isinstance(output_shape, int):
            output_shape = (output_shape,)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hiddens = hiddens

        model = []
        prev_h = np.prod(input_shape)
        for h in hiddens + [np.prod(output_shape)]:
            model.append(nn.Linear(prev_h, h))
            model.append(nn.ReLU())
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


# PixelVAE
class PixelVAE(nn.Module):
    def __init__(self, device, input_shape, latent_size, enc_dist, prior,
                 enc_conv_sizes=[(3, 64, 2)] * 2, beta=1, free_bits=0):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        self.beta = beta
        self.device = device
        self.free_bits = free_bits

        self.enc_dist = enc_dist
        self.prior = prior

        self.encoder = ConvEncoder(input_shape, get_dist_output_size(enc_dist, latent_size),
                                   enc_conv_sizes)
        self.decoder = PixelCNN(device, 2, input_shape=input_shape, conditional_size=latent_size,
                                n_layers=5, kernel_size=7)

    def encode(self, x, sample=True):
        out = self.encoder(2 * x - 1)
        if sample:
            z = self.enc_dist.sample(out)
        else:
            z = self.enc_dist.expectation(out)
        return z

    def decode(self, z):
        return self.decoder.sample(z.shape[0], cond=z).to(self.device)

    def loss(self, x):
        enc_params = self.encoder(2 * x - 1)
        z = self.enc_dist.sample(enc_params)
        self.enc_dist.set_params(enc_params)

        recon_loss = self.decoder.loss(x, cond=z)['loss'] * np.prod(self.input_shape)
        kl_loss = kl(z, self.enc_dist, self.prior)
        kl_loss = torch.clamp(kl_loss, min=self.free_bits)
        kl_loss = kl_loss.mean()

        return OrderedDict(loss=recon_loss + self.beta * kl_loss, recon_loss=recon_loss,
                           kl_loss=kl_loss)

    def sample(self, n):
        with torch.no_grad():
            z = torch.cat([self.prior.sample() for _ in range(n)], dim=0)
            return self.decoder.sample(n, cond=z)

# AF-VAE
class AFPixelVAE(nn.Module):
    def __init__(self, device, input_shape, latent_size, enc_dist,
                 enc_conv_sizes=[(3, 64, 2)] * 2, made_hidden_sizes=[512, 512], beta=1,
                 n_mades=1):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        self.n_mades = n_mades
        self.beta = beta
        self.device = device
        self.enc_dist = enc_dist

        self.mades = nn.ModuleList([MADE(latent_size, 2, hidden_size=made_hidden_sizes)
                                    for _ in range(n_mades)])
        self.encoder = ConvEncoder(input_shape, get_dist_output_size(enc_dist, latent_size),
                                   enc_conv_sizes)
        self.decoder = PixelCNN(device, 2, input_shape=input_shape, conditional_size=latent_size,
                                n_layers=5, kernel_size=7)

    def encode(self, x, sample=True):
        out = self.encoder(2 * x - 1)
        if sample:
            z = self.enc_dist.sample(out)
        else:
            z = self.enc_dist.expectation(out)
        return z

    def decode(self, z):
        return self.decoder.sample(z.shape[0], cond=z).to(self.device)

    def loss(self, x):
        enc_params = self.encoder(2 * x - 1)
        z = self.enc_dist.sample(enc_params)
        self.enc_dist.set_params(enc_params)

        recon_loss = self.decoder.loss(x, cond=z)['loss'] * np.prod(self.input_shape)
        enc_log_prob = self.enc_dist.log_prob(z)

        eps = z
        prior_log_prob = 0
        for i in range(self.n_mades):
            out = self.mades[i](eps)
            mu, log_std = out.chunk(2, dim=-1)
            log_std = torch.tanh(log_std)
            mu, log_std = mu.squeeze(-1), log_std.squeeze(-1)
            eps = eps * torch.exp(log_std) + mu
            prior_log_prob = prior_log_prob + log_std
        prior_log_prob = -0.5 * np.log(2 * np.pi) - 0.5 * eps ** 2
        prior_log_prob = prior_log_prob.sum(dim=1)

        kl_loss = (enc_log_prob - prior_log_prob).mean()

        return OrderedDict(loss=recon_loss + self.beta * kl_loss, recon_loss=recon_loss,
                           kl_loss=kl_loss)

    def sample(self, n):
        with torch.no_grad():
            z = torch.randn(n, self.latent_size).to(self.device)
            for i in range(self.n_mades):
                for j in range(self.latent_size):
                    mu, log_std = self.mades[self.n_mades - i - 1](z)[:, j].chunk(2, dim=-1)
                    log_std = torch.tanh(log_std)
                    mu, log_std = mu.squeeze(-1), log_std.squeeze(-1)
                    z[:, j] = (z[:, j] - mu) * torch.exp(-log_std) + mu
            return self.decoder.sample(n, cond=z)

# IAF-PixelVAE
class IAFPixelVAE(nn.Module):
    def __init__(self, device, input_shape, latent_size, prior,
                 enc_conv_sizes=[(3, 64, 2)] * 2, made_hidden_sizes=[512, 512],
                 embedding_size=128, beta=1, n_mades=1):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        self.n_mades = n_mades
        self.beta = beta
        self.device = device
        self.prior = prior
        self.embedding_size = embedding_size

        self.mades = nn.ModuleList([MADE(latent_size, 2, hidden_size=made_hidden_sizes,
                                         conditional_size=embedding_size)
                                    for _ in range(n_mades)])
        self.base_encoder = ConvEncoder(input_shape, (embedding_size + 2 * latent_size,),
                                        enc_conv_sizes)
        self.decoder = PixelCNN(device, 2, input_shape=input_shape, conditional_size=latent_size,
                                n_layers=5, kernel_size=7)

    def encode(self, x, sample=True, include_log_prob=False):
        eps = torch.randn(x.shape[0], self.latent_size).to(self.device)
        out = self.base_encoder(2 * x - 1)
        h = out[:, :self.embedding_size]
        base_mu, base_log_std = out[:, self.embedding_size:].chunk(2, dim=1)
        base_log_std = torch.tanh(base_log_std)

        log_prob = 0.5 * np.log(2 * np.pi) + base_log_std + 0.5 * eps ** 2
        z = base_mu + eps * base_log_std.exp()
        for i in range(self.n_mades):
            out = self.mades[i](z, cond=h)
            mu, s = out.chunk(2, dim=-1)
            mu, s = mu.squeeze(-1), s.squeeze(-1)
            s = F.logsigmoid(s)
            z = s.exp() * z + (1 - s.exp()) * mu
            log_prob = log_prob + s
        log_prob = -log_prob

        if include_log_prob:
            return z, log_prob
        else:
            return z

    def decode(self, z):
        return self.decoder.sample(z.shape[0], cond=z).to(self.device)

    def loss(self, x):
        z, enc_log_prob = self.encode(x, include_log_prob=True, sample=True)
        recon_loss = self.decoder.loss(x, cond=z)['loss'] * np.prod(self.input_shape)
        prior_log_prob = -0.5 * np.log(2 * np.pi) - 0.5 * z ** 2
        kl_loss = (enc_log_prob - prior_log_prob).mean()

        return OrderedDict(loss=recon_loss + self.beta * kl_loss, recon_loss=recon_loss,
                           kl_loss=kl_loss)

    def sample(self, n):
        with torch.no_grad():
            z = torch.randn(n, self.latent_size).to(self.device)
            return self.decoder.sample(n, cond=z)

# IWAE


# PixelCNN
class MaskConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, conditional_size=None, **kwargs):
        assert mask_type == 'A' or mask_type == 'B'
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask(mask_type)

        self.conditional_size = conditional_size
        if conditional_size:
            if len(conditional_size) == 1:
                self.cond_op = nn.Linear(conditional_size[0], self.out_channels)
            else:
                self.cond_op = nn.Conv2d(conditional_size[0], self.out_channels, 3, padding=1)

    def forward(self, input, cond=None):
        out = F.conv2d(input, self.weight * self.mask, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        if cond is not None:
            if len(self.conditional_size) == 1:
                out = out + self.cond_op(cond).unsqueeze(-1).unsqueeze(-1)
            else:
                out = out + self.cond_op(cond)
        return out

    def create_mask(self, mask_type):
        k = self.kernel_size[0]
        self.mask[:, :, :k // 2] = 1
        self.mask[:, :, k // 2, :k // 2] = 1
        if mask_type == 'B':
            self.mask[:, :, k // 2, k // 2] = 1


class PixelCNN(nn.Module):
    def __init__(self, device, d, input_shape=(1, 7, 7), kernel_size=5, n_layers=7,
                 conditional_size=None):
        super().__init__()
        assert n_layers >= 2
        if isinstance(conditional_size, int):
            conditional_size = (conditional_size,)

        model = nn.ModuleList([MaskConv2d('A', input_shape[0], 64, kernel_size,
                                          padding=kernel_size // 2,
                                          conditional_size=conditional_size), nn.ReLU()])
        for _ in range(n_layers - 2):
            model.extend([MaskConv2d('B', 64, 64, kernel_size,
                                     padding=kernel_size // 2,
                                     conditional_size=conditional_size), nn.ReLU()])
        model.append(MaskConv2d('B', 64, input_shape[0] * d, kernel_size, padding=kernel_size // 2,
                                conditional_size=conditional_size))
        self.net = model
        self.d = d
        self.device = device
        self.input_shape = input_shape

        if conditional_size:
            if len(conditional_size) == 1:
                self.cond_op = lambda x: x
            else:
                pass

    def forward(self, x, cond=None):
        out = 2 * (x.float() / (self.d - 1)) - 1
        for layer in self.net:
            if isinstance(layer, MaskConv2d):
                out = layer(out, cond=cond)
            else:
                out = layer(out)
        return out.view(x.shape[0], self.d, *self.input_shape)

    def loss(self, x, cond=None):
        return OrderedDict(loss=F.cross_entropy(self(x, cond=cond), x.long()))

    def sample(self, n, cond=None):
        samples = torch.zeros(n, *self.input_shape).to(self.device)
        with torch.no_grad():
            for r in range(self.input_shape[1]):
                for c in range(self.input_shape[2]):
                    for k in range(self.input_shape[0]):
                        logits = self(samples, cond=cond)[:, :, k, r, c]
                        logits = F.softmax(logits, dim=1)
                        samples[:, k, r, c] = torch.multinomial(logits, 1).squeeze(-1)
        return samples.cpu()


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
        quantized = self.embedding(encoding_indices).permute(0, 3, 1, 2).contiguous()

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
        latents = self.codebook.embedding(latents).permute(0, 3, 1, 2).contiguous()
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

# Gumbel-Softmax VAE: https://github.com/YongfeiYan/Gumbel_Softmax_VAE/
def sample_gumbel(shape, eps=1e-20):
    U = torch.clamp(torch.rand(shape), min=1e-5, max=1 - 1e-5).cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, latent_dim, categorical_dim, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y.view(-1, latent_dim * categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_dim * categorical_dim)

class GumbelVAE(nn.Module):
    ANNEAL_RATE = 3e-5
    MIN_TEMP = 0.5

    def __init__(self, latent_dim, categorical_dim, hard=False):
        super().__init__()

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim * categorical_dim)

        self.fc4 = nn.Linear(latent_dim * categorical_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)

        self.relu = nn.ReLU()

        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.hard = hard

        self.temp = 1
        self.counter = 0

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))

    def decode(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return torch.sigmoid(self.fc6(h5))

    def forward(self, x):
        q = self.encode(x.view(-1, 784))
        q_y = q.view(q.size(0), self.latent_dim, self.categorical_dim)
        z = gumbel_softmax(q_y, self.temp, self.latent_dim, self.categorical_dim, self.hard)

        self.temp = max(self.temp * np.exp(-GumbelVAE.ANNEAL_RATE * self.counter), GumbelVAE.MIN_TEMP)
        self.counter += 1
        return self.decode(z), F.softmax(q_y, dim=-1).reshape(*q.size())

    def loss(self, x):
        recon_batch, qy = self(x)
        recon_loss = F.binary_cross_entropy(recon_batch, x.view(-1, 784), reduction='none').sum(-1).mean()
        log_ratio = torch.log(qy * self.categorical_dim + 1e-20)
        kl_loss = torch.sum(qy * log_ratio, dim=-1).mean()
        return OrderedDict(loss=recon_loss + kl_loss, recon_loss=recon_loss, kl_loss=kl_loss)

    def sample(self, n):
        with torch.no_grad():
            z = torch.zeros(n * self.latent_dim, self.categorical_dim).float()
            idxs = torch.randint(high=self.categorical_dim, size=(n * self.latent_dim,))
            z.scatter_(1, idxs.unsqueeze(1), 1)
            z = z.view(n, self.latent_dim * self.categorical_dim).cuda()
            return self.decode(z).view(n, 1, 28, 28).cpu()

    def train(self):
        super().train()
        self.temp = 1.0
        self.counter = 0


    def eval(self):
        super().train()
        self.temp = 1.0
        self.counter = 0
