import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets
from torchvision import transforms

from deepul_helper.models import ConvVAE, FullyConnectedVAE
from deepul_helper.trainer import train_epochs
from deepul_helper.distributions import Normal, Bernoulli
from deepul_helper.visualize import visualize_batch

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dset = datasets.MNIST('data', transform=transform, train=True, download=True)
test_dset = datasets.MNIST('data', transform=transform, train=False, download=True)

train_loader = data.DataLoader(train_dset, batch_size=128, shuffle=True, pin_memory=True,
                               num_workers=2)
test_loader = data.DataLoader(test_dset, batch_size=128, pin_memory=True, num_workers=2)

device = torch.device('cuda')
latent_size = 8
enc_dist, dec_dist = Normal(), Normal(use_mean=True)
prior = Normal(torch.cat((torch.zeros(1, latent_size), torch.ones(1, latent_size)), dim=1).to(device))
mnist_vae = FullyConnectedVAE((1, 28, 28), 8, enc_dist, dec_dist, prior,
                              enc_hidden_sizes=[512, 512], dec_hidden_sizes=[512, 512],
                              output_activation=torch.tanh).to(device)
train_epochs(mnist_vae, train_loader, test_loader, device, dict(epochs=10, lr=1e-3))

samples = mnist_vae.sample(100, decoder_noise=False)
visualize_batch(samples * 0.5 + 0.5, nrow=10, title='Samples')

with torch.no_grad():
    x = next(iter(train_loader))[0][:50].to(device)
    z = mnist_vae.encode(x)
    x_recon = mnist_vae.decode(z, sample=False)
    images = torch.stack((x, x_recon), dim=1).view(-1, 1, 28, 28).cpu()
visualize_batch(images * 0.5 + 0.5, nrow=10, title='Reconstructions')

with torch.no_grad():
    x = next(iter(train_loader))[0][:20].to(device)
    z1, z2 = mnist_vae.encode(x).chunk(2, dim=0)
    zs = [z1 * (1 - alpha) + z2 * alpha for alpha in np.linspace(0, 1, 10)]
    zs = torch.stack(zs, dim=1).view(-1, latent_size)
    xs = mnist_vae.decode(zs, sample=False).cpu()
visualize_batch(xs * 0.5 + 0.5, nrow=10, title='Interpolations')




