import torch
import torch.utils.data as data

from deepul_helper.data import sample_smiley_data
from deepul_helper.visualize import plot_scatter_2d
from deepul_helper.models import FullyConnectedVAE
from deepul_helper.trainer import train_epochs
from deepul_helper.distributions import Normal

train_data, train_labels = sample_smiley_data(100000)
test_data, test_labels = sample_smiley_data(25000)
device = torch.device('cuda')

enc_dist, dec_dist = Normal(), Normal()
prior = Normal(torch.cat((torch.zeros(1, 2), torch.ones(1, 2)), dim=1).to(device))
vae = FullyConnectedVAE(2, 2, enc_dist, dec_dist, prior,
                        enc_hidden_sizes=[128, 128], dec_hidden_sizes=[128, 128],
                        beta=1).to(device)

plot_scatter_2d(train_data[:10000], title='Train Data', labels=train_labels[:10000])

train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = data.DataLoader(test_data, batch_size=128, shuffle=False)

train_epochs(vae, train_loader, test_loader, device, dict(epochs=20, lr=1e-3))

samples = vae.sample(1000).numpy()
plot_scatter_2d(samples, title='Samples')

x = torch.FloatTensor(train_data[:10000]).to(device)
with torch.no_grad():
    z = vae.encode(x)
    x_recon = vae.decode(z, sample=False)
z, x_recon = z.cpu().numpy(), x_recon.cpu().numpy()
plot_scatter_2d(x_recon, title='Reconstruction', labels=train_labels[:10000])
plot_scatter_2d(z, title='Latent Space', labels=train_labels[:10000])

