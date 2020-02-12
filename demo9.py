import torch
import torch.utils.data as data
from torchvision import datasets
from torchvision import transforms

from deepul_helper.models import PixelVAE, AFPixelVAE, IAFPixelVAE
from deepul_helper.trainer import train_epochs
from deepul_helper.visualize import visualize_batch
from deepul_helper.distributions import Normal

def run_demo9(pixel_vae):
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: (x > 0.5).float()
    ])
    train_dset = datasets.MNIST('data', transform=transform, train=True, download=True)
    test_dset = datasets.MNIST('data', transform=transform, train=False, download=True)

    train_loader = data.DataLoader(train_dset, batch_size=128, shuffle=True, pin_memory=True,
                                   num_workers=2)
    test_loader = data.DataLoader(test_dset, batch_size=128, pin_memory=True, num_workers=2)

    train_epochs(pixel_vae, train_loader, test_loader, device, dict(epochs=10, lr=1e-3))

    # Reconstructions
    with torch.no_grad():
        x = next(iter(test_loader))[0][:50].to(device)
        z = pixel_vae.encode(x)
        x_recon = pixel_vae.decode(z)
        images = torch.stack((x, x_recon), dim=1).view(-1, 1, 28, 28).cpu()
    visualize_batch(images, nrow=10, title='Reconstructions')


device = torch.device('cuda')
latent_size = 8
enc_dist = Normal()
prior = Normal(torch.cat((torch.zeros(1, latent_size), torch.ones(1, latent_size)), dim=1).to(device))

pixel_vae = PixelVAE(device, (1, 28, 28), latent_size, enc_dist, prior,
                     enc_conv_sizes=((3, 64, 2), (3, 64, 2))).to(device)
run_demo9(pixel_vae)

pixel_vae_free = PixelVAE(device, (1, 28, 28), latent_size, enc_dist, prior,
                         enc_conv_sizes=((3, 64, 2), (3, 64, 2)),
                         free_bits=3).to(device)
run_demo9(pixel_vae_free)


af_vae = AFPixelVAE(device, (1, 28, 28), latent_size, enc_dist,
                       enc_conv_sizes=((3, 64, 2), (3, 64, 2)), made_hidden_sizes=[512, 512]).to(device)
run_demo9(af_vae)

iaf_vae = IAFPixelVAE(device, (1, 28, 28), latent_size, prior,
                      enc_conv_sizes=((3, 64, 2), (3, 64, 2)), made_hidden_sizes=[512, 512],
                      embedding_size=128).to(device)
run_demo9(iaf_vae)
