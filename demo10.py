import torch
import torch.utils.data as data
from torchvision import datasets
from torchvision import transforms

from deepul_helper.models import VectorQuantizedVAE, PixelCNN, GumbelVAE
from deepul_helper.trainer import train_epochs
from deepul_helper.visualize import visualize_batch

seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])
train_dset = datasets.MNIST('data', transform=transform, train=True, download=True)
test_dset = datasets.MNIST('data', transform=transform, train=False, download=True)

train_loader = data.DataLoader(train_dset, batch_size=128, shuffle=True, pin_memory=True,
                               num_workers=2)
test_loader = data.DataLoader(test_dset, batch_size=128, pin_memory=True, num_workers=2)
device = torch.device('cuda')

# VQ-VAE
# vqvae = VectorQuantizedVAE(64, 4, 1).to(device)
# train_epochs(vqvae, train_loader, test_loader, device, dict(epochs=10, lr=1e-3))
#
# with torch.no_grad():
#     x = next(iter(test_loader))[0][:50].to(device)
#     z = vqvae.encode_code(x)
#     x_recon = vqvae.decode_code(z)
#     images = torch.stack((x, x_recon), dim=1).view(-1, 1, 28, 28).cpu()
# visualize_batch(images * 0.5 + 0.5, nrow=10, title='Reconstructions')
#
# def construct_z_dset(vqvae, data_loader):
#     zs = []
#     for x, _ in data_loader:
#         with torch.no_grad():
#             x = x.to(device)
#             zs.append(vqvae.encode_code(x).unsqueeze(1).cpu())
#     zs = torch.cat(zs, dim=0)
#     return zs
#
# train_z_dset = construct_z_dset(vqvae, train_loader)
# test_z_dset = construct_z_dset(vqvae, test_loader)
# train_z_loader = data.DataLoader(train_z_dset, batch_size=128, shuffle=True, pin_memory=True,
#                                  num_workers=2)
# test_z_loader = data.DataLoader(test_z_dset, batch_size=128, pin_memory=True, num_workers=2)
#
# pixelcnn_prior = PixelCNN(device, 4, n_layers=10).to(device)
# train_epochs(pixelcnn_prior, train_z_loader, test_z_loader, device, dict(epochs=20, lr=1e-3))
#
# with torch.no_grad():
#     samples = pixelcnn_prior.sample(100).squeeze(1).long().to(device)
#     samples = vqvae.decode_code(samples).cpu()
# visualize_batch(samples * 0.5 + 0.5, nrow=10, title='Samples')

# Gumbel Softmax VAE
gumbel_vae = GumbelVAE(30, 10).to(device)
train_epochs(gumbel_vae, train_loader, test_loader, device, dict(epochs=20, lr=1e-3))

gumbel_vae.temp = 0.1
gumbel_vae.hard = True
with torch.no_grad():
    x = next(iter(test_loader))[0][:50].to(device)
    x_recon, _ = gumbel_vae(x)
    x_recon = x_recon.view(-1, 1, 28, 28)
    xs = torch.stack((x, x_recon), dim=1).view(-1, 1, 28, 28).cpu()
visualize_batch(xs, nrow=10, title='Reconstructions')

samples = gumbel_vae.sample(100)
visualize_batch(samples, nrow=10, title='Samples')

