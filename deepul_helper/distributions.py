import numpy as np
import torch
from torch.distributions import Normal

def normal_log_prob(x, mu, log_stddev):
    """
    Returns log probability of x according to given params to diagonal gaussian
    :param x: data of shape (N, *)
    :param mu:  mu of shape (N, *)
    :param log_stddev:  log_stddev of shape (N, *)
    :return: log_probs of x of shape (N,)
    """
    dist = Normal(loc=mu, scale=log_stddev.exp())
    return dist.log_prob(x).view(x.shape[0], -1).sum(-1)

    log_prob = 0.5 * np.log(2 * np.pi)
    log_prob = log_prob + log_stddev + (x - mu) ** 2 * torch.exp(-2 * log_stddev) * 0.5
    log_prob = log_prob.view(log_prob.shape[0], -1).sum(-1)
    return -log_prob


def normal_kl(mu1, log_stddev1, mu2, log_stddev2):
    """
    p ~ N(mu1, log_stddev1)
    q ~ N(mu2, log_stddev2)

    Return KL(p || q) of shape (N,)
    """
    kl = log_stddev2 - log_stddev1 - 0.5
    kl = kl + (torch.exp(2 * log_stddev1) + (mu1 - mu2) ** 2) * 0.5 * torch.exp(-2 * log_stddev2)
    return kl.view(kl.shape[0], -1).sum(-1)