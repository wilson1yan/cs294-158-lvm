import numpy as np
import torch
import torch.nn.functional as F


def kl(z, dist1, dist2):
    if isinstance(dist1, Normal) and isinstance(dist2, Normal):
        params1, params2 = dist1.get_params(), dist2.get_params()
        mu1, log_stddev1 = params1.chunk(2, dim=1)
        mu2, log_stddev2 = params2.chunk(2, dim=1)
        kl = log_stddev2 - log_stddev1 - 0.5
        kl = kl + (torch.exp(2 * log_stddev1) + (mu1 - mu2) ** 2) * 0.5 * torch.exp(-2 * log_stddev2)
        return kl.view(kl.shape[0], -1).sum(-1)
    else:
        return dist1.log_prob(z) - dist2.log_prob(z)


class Distribution(object):
    def __init__(self, params=None):
        self.params = params

    def log_prob(self, x, params=None):
        raise NotImplementedError()

    def expectation(self, params=None):
        raise NotImplementedError()

    def sample(self, params=None):
        raise NotImplementedError()

    def set_params(self, params):
        self.params = params

    def get_params(self, params=None):
        if params is None:
            params = self.params
        assert params is not None
        return params


class Normal(Distribution):
    def __init__(self, params=None, use_mean=False, tanh_std_dev=False):
        super().__init__(params=params)
        self.use_mean = use_mean
        self.tanh_std_dev = tanh_std_dev

    def log_prob(self, x, params=None):
        params = self.get_params(params)
        mu, log_stddev = params.chunk(2, dim=1)
        if self.tanh_std_dev:
            log_stddev = torch.tanh(log_stddev)

        if self.use_mean:
            return -F.mse_loss(mu, x, reduction='none').view(x.shape[0], -1).sum(-1)

        log_prob = 0.5 * np.log(2 * np.pi)
        log_prob = log_prob + log_stddev + (x - mu) ** 2 * torch.exp(-2 * log_stddev) * 0.5
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(-1)
        return -log_prob

    def expectation(self, params=None):
        params = self.get_params(params)
        return params.chunk(2, dim=1)[0]

    def sample(self, params=None):
        params = self.get_params(params)
        mu, log_stddev = params.chunk(2, dim=1)
        eps = torch.randn_like(mu)
        return mu + eps * log_stddev.exp()

class Bernoulli(Distribution):
    def log_prob(self, x, params=None):
        params = self.get_params(params)
        return -F.binary_cross_entropy_with_logits(params, x, reduction='none').view(x.shape[0], -1).sum(-1)

    def expectation(self, params=None):
        return torch.sigmoid(self.get_params(params))

    def sample(self, params=None):
        params = self.get_params(params)
        return torch.bernoulli(torch.sigmoid(params))


def get_dist_output_size(dist, var_shape, flattened=False):
    if flattened or isinstance(var_shape, int) or len(var_shape) == 1:
        flattened_size = np.prod(var_shape)
        if isinstance(dist, Normal):
            return (2 * flattened_size,)
        elif isinstance(dist, Bernoulli):
            return (flattened_size,)
        else:
            raise Exception('Invalid dist')
    else:
        assert len(var_shape) == 3
        if isinstance(dist, Normal):
            return (var_shape[0] * 2,) + var_shape[1:]
        elif isinstance(dist, Bernoulli):
            return var_shape
