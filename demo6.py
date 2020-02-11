import numpy as np
import torch
import torch.utils.data as data
from deepul_helper.visualize import plot_scatter_2d
from deepul_helper.models import FullyConnectedVAE, FullyConnectedIAFVAE
from deepul_helper.trainer import train_epochs
from deepul_helper.distributions import Normal

def demo6(vae, train_data, device):
    # train_data = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], dtype='float32')

    train_loader = data.DataLoader(train_data, batch_size=4, shuffle=True)
    train_epochs(vae, train_loader, None, device, dict(epochs=400, lr=1e-3))

    points, labels = [], []
    with torch.no_grad():
        for i in range(4):
            x = torch.FloatTensor(train_data[[i]]).to(device)
            x = x.repeat(1000, 1)
            z = vae.encode(x, sample=True)
            points.append(z.cpu().numpy())
            labels.append([i] * 1000)
    points = np.concatenate(points, axis=0)
    labels = np.concatenate(labels, axis=0)
    plot_scatter_2d(points, title='Learned Posterior', labels=labels)

# Standard VAE
# train_data = np.random.randn(4, 10).astype('float32')
train_data = 5 * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype='float32')
device = torch.device('cpu')

# enc_dist, dec_dist = Normal(), Normal()
# prior = Normal(torch.cat((torch.zeros(1, 2), torch.ones(1, 2)), dim=1).to(device))
# vae = FullyConnectedVAE(device, train_data.shape[1], 2, enc_dist, dec_dist, prior,
#                         enc_hidden_sizes=[32, 32], dec_hidden_sizes=[32]).to(device)
# demo6(vae, train_data, device)

dec_dist = Normal()
iaf_vae = FullyConnectedIAFVAE(device, train_data.shape[1], 16, 2, dec_dist,
                               base_hidden_sizes=[32], made_hidden_sizes=[64, 64],
                               dec_hidden_sizes=[32]).to(device)
demo6(iaf_vae, train_data, device)