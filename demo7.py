import torch
import torch.utils.data as data
from torchvision import datasets
from torchvision import transforms

from deepul_helper.models import ConvVAE
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
mnist_vae = ConvVAE((1, 28, 28), latent_size, enc_dist, dec_dist, prior,
                    enc_conv_sizes=[(5, 64, 2), (5, 64, 2), (5, 64, 1)],
                    dec_base_size=(1, 7, 7),
                    dec_conv_sizes=[(4, 64, 2, 1), (4, 64, 2, 1), (5, 64, 1, 2)]).to(device)
train_epochs(mnist_vae, train_loader, test_loader, device, dict(epochs=20, lr=1e-3))

samples = mnist_vae.sample(100, decoder_noise=False)
visualize_batch(samples * 0.5 + 0.5, nrow=10, title='Samples')

with torch.no_grad():
    x = next(iter(train_loader))[0][:50].to(device)
    z = mnist_vae.encode(x)
    x_recon = mnist_vae.decode(z, sample=False)
    images = torch.stack((x, x_recon), dim=1).view(-1, 1, 28, 28).cpu()
visualize_batch(images * 0.5 + 0.5, nrow=10, title='Reconstructions')



