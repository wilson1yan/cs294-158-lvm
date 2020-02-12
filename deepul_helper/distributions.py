import numpy as np
import torch
import torch.nn.functional as F


def kl(z, dist1, dist2):
    if isinstance(dist1, Normal) and isinstance(dist2, Normal):
        mu1, log_var1 = dist1.get_mu_std()
        mu2, log_var2 = dist2.get_mu_std()
        kl = 0.5 * log_var2 - 0.5 * log_var1 - 0.5
        kl = kl + (torch.exp(log_var1) + (mu1 - mu2) ** 2) * 0.5 * torch.exp(-log_var2)
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
    def __init__(self, params=None, use_mean=False, min_std_dev=None):
        super().__init__(params=params)
        self.use_mean = use_mean
        self.min_std_dev = min_std_dev

    def get_mu_std(self):
        params = self.get_params(None)
        mu, log_var = params.chunk(2, dim=1)
        if self.min_std_dev is not None:
            log_var = torch.log(log_var.exp() + 1 + self.min_std_dev)
        return mu, log_var


    def log_prob(self, x, params=None):
        params = self.get_params(params)
        mu, log_var = params.chunk(2, dim=1)
        if self.min_std_dev is not None:
            log_var = torch.log(log_var.exp() + 1 + self.min_std_dev)

        if self.use_mean:
            return -F.mse_loss(mu, x, reduction='none').view(x.shape[0], -1).sum(-1)

        log_prob = 0.5 * np.log(2 * np.pi)
        log_prob = log_prob + 0.5 * log_var + (x - mu) ** 2 * torch.exp(-log_var) * 0.5
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(-1)
        return -log_prob

    def expectation(self, params=None):
        params = self.get_params(params)
        return params.chunk(2, dim=1)[0]

    def sample(self, params=None):
        params = self.get_params(params)
        mu, log_var = params.chunk(2, dim=1)
        eps = torch.randn_like(mu)
        return mu + eps * (0.5 * log_var).exp()

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
