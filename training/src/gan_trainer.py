import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import Subset
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from .datasets import WikiArtDataset
from .gan_models import *


def generate_latent_point(n_samples, latent_dim):
    x = torch.randn(n_samples, latent_dim)
    return x

def generate_false_samples(g_model, device, batch_size=500, latent_dim=100):
    X = generate_latent_point(batch_size, latent_dim)
    # X = X.view(batch_size, latent_dim, 1, 1)
    # print(f'Latent Points: {X.shape}')
    X = X.to(device)
    X = g_model(X)
    y = torch.zeros((batch_size, 1))
    X = X.to(device)
    y = y.to(device)
    # print(f'False sample: {X.shape}, y: {y.shape}')
    return X, y


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_plot(samples, epoch, n=3):
    # plot images
    plt.figure(figsize=(32, 32))
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        # print(samples[i])
        # print(f'samples shape: {samples[i, :, :, :].shape}')
        img = samples[i, :, :, :].detach().cpu().permute(1, 2, 0).numpy()
        normalized_img = (img + 1) / 2
        # print(img.shape)
        plt.imshow(normalized_img)
    # save plot to file
    filename = './generated_plot_e%03d.png' % (epoch + 1)
    plt.savefig(filename)
    plt.close()

def summarize_performance(epoch, g_model, d_model, test_loader, device, batch_size, latent_dim=100):
    X_false, y_false = generate_false_samples(g_model, device, batch_size, latent_dim)
    X_false = X_false.to(device)
    y_false = y_false.to(device)

    # Test model
    # test_model(X_false, batch_size, d_model, device, epoch, test_loader, y_false)

    # Plot images
    save_plot(X_false, epoch)


def test_model(X_false, batch_size, d_model, device, epoch, test_loader, y_false):
    avg_true_acc = 0.
    with torch.no_grad():
        for b, data in enumerate(test_loader):
            test_inputs, _ = data
            test_inputs = test_inputs.to(device)
            # test_labels = test_labels.to(device)

            test_labels = torch.ones((batch_size, 1), dtype=torch.float).to(device)

            test_outputs = d_model(test_inputs)

            # Accuracy
            predicted = test_outputs > 0.5  # Threshold logits at 0 for classification
            correct = (predicted == test_labels).float().sum()

            acc = correct / test_outputs.size(0)

            avg_true_acc += acc.cpu()
        avg_true_acc = avg_true_acc / len(test_loader)
    false_outputs = d_model(X_false)
    false_predictd = (false_outputs > 0.5).to(device)
    false_correct = (false_predictd == y_false).float().sum()
    false_acc = false_correct / len(X_false)
    print(f'\tEpoch[{epoch}] True acc: {avg_true_acc:.4f}, False acc: {false_acc:.4f}')


def discriminator_loss(output_true, output_fake):
    true_loss = F.binary_cross_entropy(output_true, torch.ones_like(output_true))
    fake_loss = F.binary_cross_entropy(output_fake, torch.zeros_like(output_fake))
    total_loss = true_loss + fake_loss
    return total_loss

def generator_loss(output_fake):
    generated_loss = F.binary_cross_entropy(output_fake, torch.ones_like(output_fake))
    return generated_loss



def train_gan(g_model, d_model, gan_model, train_loader, device, epochs=100, batch_size=500, latent_dim=100):
    optimizer_g = torch.optim.Adam(g_model.parameters(), lr=GEN_LR, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(d_model.parameters(), lr=DIS_LR, betas=(0.5, 0.999))
    print(device)
    for epoch in range(epochs):
        for b, batch_dataset in enumerate(train_loader):
            X_true, _ = batch_dataset
            X_true = X_true.to(device)

            y_true = torch.ones((batch_size, 1), dtype=torch.float)
            y_true = y_true.to(device)

            # Phase 1: Discriminating: Focusing on updating discriminator
            ## Generate false samples with a trained generator

            X_false, y_false = generate_false_samples(g_model, device, batch_size, latent_dim)

            # print(f'\tY shape, true: {y_true.shape}, false: {y_false.shape}')
            optimizer_d.zero_grad()
            d_true = d_model(X_true)
            # print(f'D model output shape: {d_true.shape}')

            true_loss = F.binary_cross_entropy(d_true, torch.ones_like(d_true))
            true_loss.backward()
            d_x = d_true.mean().item() # Loss of D(x)

            d_fake = d_model(X_false.detach())
            fake_loss = F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
            fake_loss.backward()
            d_g_z1 = d_fake.mean().item()

            d_loss = true_loss + fake_loss
            # if (b + 1) % 2 == 0:
            # d_loss.backward()
            optimizer_d.step()

            # Phase 2: Polishing fake examples: Focusing on updating generator
            # X_false, y_false = generate_false_samples_v2(g_model, device, batch_size * 2, latent_dim)
            optimizer_g.zero_grad()
            g_fake = d_model(X_false)
            g_loss = F.binary_cross_entropy(g_fake, torch.ones_like(g_fake))
            g_loss.backward()
            d_g_z2 = g_fake.mean().item()
            optimizer_g.step()

            print(f'Epoch[{epoch}], Batch[{b}] Loss G: {g_loss:.4f}, Loss D: {d_loss:.4f}, D(x): {d_x:.4f}, D(G(z1)): {d_g_z1:.4f}, D(G(z2)): {d_g_z2:.4f}')

        if (epoch + 1) % 2 == 0:
            # Plot performance every 5 epoch
            summarize_performance(epoch, g_model, d_model, None, device, batch_size, latent_dim)

        if (epoch + 1) % 50 == 0:
            save_model(epoch, g_model)


def find_device():
    device = 'cpu'
    # Check if MPS is supported and available
    if torch.backends.mps.is_available():
        print("MPS is available on this device.")
        device = torch.device("mps")  # Use MPS device
    else:
        print("MPS not available, using CPU instead.")
        device = torch.device("cpu")  # Fallback to CPU
    return device

def save_model(epoch, g_model):
    # save the generator model tile file
    filename = './generator_model_%03d.pt' % (epoch + 1)
    torch.save(g_model.state_dict(), filename)

def custom_collate_fn(batch):
    batch = list(filter(None, batch))  # Remove None items
    if not batch:
        return torch.tensor([]), torch.tensor([])  # Return empty tensors if batch is empty
    return torch.utils.data.dataloader.default_collate(batch)

def run() :
    device = find_device()

    img_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((INPUT_DIM, INPUT_DIM)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    genra_dataset = WikiArtDataset(root_dir='../wikiart/', category=CATEGORY_GENRE, transform=img_transforms,
                                   label_filters=LABEL_FILTERS)
    dataloader = DataLoader(genra_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=TORCH_WORKERS, collate_fn=custom_collate_fn)

    d_model = DCGANDiscriminatorNet()
    d_model.to(device)
    d_model.apply(weights_init)

    g_model = DCGANGeneratorNet()
    g_model.to(device)
    g_model.apply(weights_init)

    gan_model = GAN(g_model, d_model)
    gan_model.to(device)

    # summarize_performance(10, g_model, d_model, None, device, BATCH_SIZE, LATENT_DIM)
    epoch = EPOCHS
    train_gan(g_model, d_model, gan_model, dataloader, device, epochs=epoch, batch_size=BATCH_SIZE, latent_dim=LATENT_DIM)
    save_model(epoch, g_model)


if __name__ == '__main__':
    Image.MAX_IMAGE_PIXELS = None  # Completely removes the limit
    # or set to a higher limit, for example:
    Image.MAX_IMAGE_PIXELS = 100_000_000  # set to a more suitable limit
    start_ts = time.time()
    print(f'Training started ....')
    run()
    print(f'Training finished ... ({(time.time() - start_ts)/1000} s)')