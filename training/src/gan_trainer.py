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
from datetime import datetime


import wandb

from src.datasets import WikiArtDataset
from src.configs import train_configs
from src.gan_models import *


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



def train_gan(g_model, d_model, train_loader, device, epochs=100, batch_size=500, latent_dim=100, loss_type='bce'):
    optimizer_g = torch.optim.Adam(g_model.parameters(), lr=train_configs['GEN_LR'], betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(d_model.parameters(), lr=train_configs['DIS_LR'], betas=(0.5, 0.999))
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
            d_fake = d_model(X_false.detach())
            d_x = d_true.mean().item()  # Loss of D(x)
            d_g_z1 = d_fake.mean().item()

            if loss_type == 'bce':
                true_loss = F.binary_cross_entropy(d_true, torch.ones_like(d_true))
                true_loss.backward()
                fake_loss = F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
                fake_loss.backward()
                d_loss = true_loss + fake_loss
                # d_loss.backward()
            else:
                d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - d_true)) + \
                         torch.mean(nn.ReLU(inplace=True)(1 + d_fake))
                # if (b + 1) % 2 == 0:
                d_loss.backward()
            optimizer_d.step()

            # Phase 2: Polishing fake examples: Focusing on updating generator
            # X_false, y_false = generate_false_samples_v2(g_model, device, batch_size * 2, latent_dim)
            optimizer_g.zero_grad()
            g_fake = d_model(X_false)

            if loss_type == 'bce':
                g_loss = F.binary_cross_entropy(g_fake, torch.ones_like(g_fake))
            else:
                g_loss = -torch.mean(g_fake)
            g_loss.backward()
            optimizer_g.step()
            d_g_z2 = g_fake.mean().item()

            print(f'Epoch[{epoch}], Batch[{b}] Loss G: {g_loss:.4f}, Loss D: {d_loss:.4f}, D(x): {d_x:.4f}, D(G(z1)): {d_g_z1:.4f}, D(G(z2)): {d_g_z2:.4f}')
            wandb.log({'G_loss': g_loss,
                       'D_loss': d_loss,
                       'D(x)': d_x,
                       'D(G(z1))': d_g_z1,
                       'D(G(z2))': d_g_z2
                       })
        if (epoch + 1) % 1 == 0:
            # Plot performance every 5 epoch
            summarize_performance(epoch, g_model, d_model, None, device, batch_size, latent_dim)

        if (epoch + 1) % 30 == 0:
            save_model(epoch, g_model, d_model)


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

def save_model(epoch, g_model, d_model):
    # save the generator model tile file
    filename = './generator_model_%03d.pt' % (epoch + 1)
    torch.save(g_model.state_dict(), filename)

    filename = './discriminator_model_%03d.pt' % (epoch + 1)
    torch.save(d_model.state_dict(), filename)

def custom_collate_fn(batch):
    batch = list(filter(None, batch))  # Remove None items
    if not batch:
        return torch.tensor([]), torch.tensor([])  # Return empty tensors if batch is empty
    return torch.utils.data.dataloader.default_collate(batch)


def getGANModelInstances(train_configs):
    arch = train_configs['ARCH']
    if arch == 'DCGAN':
        return DCGANGeneratorNet(), DCGANDiscriminatorNet()
    elif arch == 'SNDCGAN':
        return DCGANGeneratorNet(), SNDCGANDiscriminatorNet()
    elif arch =='SNDCGAN64':
        return DCGANGeneratorNet64(), SNDCGANDiscriminatorNet64()
    elif arch == 'DCGAN256':
        return DCGANGeneratorNet256(), DCGANDiscriminatorNet256()
    elif arch == 'SNGAN':
        return SNGANGeneratorNet(), SNGANDiscriminatorNet()
    elif arch == 'SNGAN128':
        return SNGANGeneratorNet128(), SNGANDiscriminatorNet128()
    elif arch == 'ResGAN':
        return ResGANGeneratorNet(), ResGANDiscriminatorNet()
    elif arch == 'ResGAN128':
        return ResGANGeneratorNet128(), ResGANDiscriminatorNet128()
    else:
        raise ValueError(f"Unknow architectures {arch}")


def run() :
    device = find_device()

    img_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((train_configs['INPUT_DIM'], train_configs['INPUT_DIM'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    genra_dataset = WikiArtDataset(root_dir='../wikiart/', category=train_configs['CATEGORY_GENRE'], transform=img_transforms,
                                   label_filters=train_configs['LABEL_FILTERS'])
    dataloader = DataLoader(genra_dataset, batch_size=train_configs['BATCH_SIZE'],
                            shuffle=True, num_workers=train_configs['TORCH_WORKERS'], collate_fn=custom_collate_fn)

    use_trained_model = False
    # use_model = None
    if use_trained_model:
        g_model_path = 'modes/generator_model_resnet_151.pt'
        g_model = SNGANGeneratorNet()  # Ensure the class is defined or imported
        g_model.load_state_dict(torch.load(g_model_path))
        g_model.to(device)
        g_model.eval()
    else:
        g_model, d_model = getGANModelInstances(train_configs)
        print(f'G-Model: \n{g_model}')
        print(f'D-Model: \n{d_model}')
        # d_model = SNGANDiscriminatorNet()
        d_model.to(device)
        d_model.apply(weights_init)

        # g_model = SNGANGeneratorNet()
        g_model.to(device)
        g_model.apply(weights_init)


    # == Login wandb to enable data collection ==
    expr_key = datetime.now().strftime('%Y%m%d%H%M')
    arch = train_configs['ARCH']
    expr_name = f"{arch}-{train_configs['CATEGORY_GENRE']}-{expr_key}"
    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project="XArtist",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=expr_name,
        # Track hyperparameters and run metadata
        config={**train_configs})
    # ===Wandb init finished===


    # summarize_performance(10, g_model, d_model, None, device, BATCH_SIZE, LATENT_DIM)
    epoch = train_configs['EPOCHS']
    train_gan(g_model, d_model,  dataloader, device, epochs=epoch, batch_size=train_configs['BATCH_SIZE'], latent_dim=train_configs['LATENT_DIM'], loss_type=train_configs['LOSS_FN'])
    save_model(epoch, g_model, d_model)


