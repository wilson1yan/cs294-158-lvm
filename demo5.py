import numpy as np
import torch
import torch.utils.data as data

from deepul_helper.models import FullyConnectedVAE
from deepul_helper.trainer import train_epochs
from deepul_helper.distributions import Normal, Bernoulli
from deepul_helper.visualize import plot_scatter_2d
from deepul_helper.data import sample_diag_guass_data, sample_cov_gauss_data

# Autoencoding a Single bit
def train_bern_vae(train_data):
    device = torch.device('cpu')
    enc_dist, dec_dist = Normal(), Bernoulli()
    prior = Normal(torch.FloatTensor([[0, 1]]))
    vae = FullyConnectedVAE(device, train_data.shape[1], 1, enc_dist, dec_dist, prior,
                            enc_hidden_sizes=[32], dec_hidden_sizes=[32],
                            beta=1)
    data_loader = data.DataLoader(train_data, batch_size=1, shuffle=True)
    train_epochs(vae, data_loader, None, device, dict(epochs=1000, lr=2e-4))

train_bern_vae(np.array([[0.], [1.]], dtype='float32'))
train_bern_vae(np.array([[0., 0., 0., 0., 0.], [1., 1., 1., 1., 1.]], dtype='float32'))

# Posterior Collapse on Gaussian data

def train_gauss_vae(data_fn):
    train_data = data_fn(10000)
    test_data = data_fn(2500)
    device = torch.device('cuda')

    enc_dist, dec_dist = Normal(), Normal()
    prior = Normal(torch.cat((torch.zeros(1, 2), torch.ones(1, 2)), dim=1).to(device))
    vae = FullyConnectedVAE(device, 2, 2, enc_dist, dec_dist, prior,
                            enc_hidden_sizes=[128, 128], dec_hidden_sizes=[128, 128],
                            beta=1).to(device)

    plot_scatter_2d(train_data[:10000], title='Train Data')

    train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=128, shuffle=False)

    train_epochs(vae, train_loader, test_loader, device, dict(epochs=20, lr=1e-3))

    samples = vae.sample(1000, decoder_noise=True).numpy()
    plot_scatter_2d(samples, title='Samples (with decoder noise)')
    samples = vae.sample(1000, decoder_noise=False).numpy()
    plot_scatter_2d(samples, title='Samples (without decoder noise)')

train_gauss_vae(sample_diag_guass_data)
train_gauss_vae(sample_cov_gauss_data)
