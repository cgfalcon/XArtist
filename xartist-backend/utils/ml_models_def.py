
import time

import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
import torch
from torch.utils.data import Subset
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import *

class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()

        self.gan = nn.Sequential(
            generator,
            discriminator
        )


    def forward(self, x):
        return self.gan(x)

class DCGANDiscriminatorNet64(nn.Module):
    '''64 * 64'''

    def __init__(self):
        super(DCGANDiscriminatorNet64, self).__init__()
        self.cons_layers = nn.Sequential(
            nn.Conv2d(INPUT_CHN, DIS_FILTERS, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DIS_FILTERS),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*2) x 32 x 32
            nn.Conv2d(DIS_FILTERS, DIS_FILTERS * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DIS_FILTERS * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*4) x 16 x 16
            nn.Conv2d(DIS_FILTERS * 2, DIS_FILTERS * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DIS_FILTERS * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*8) x 8 x 8
            nn.Conv2d(DIS_FILTERS * 4, DIS_FILTERS * 8, 8, 2, 1, bias=False),
            nn.BatchNorm2d(DIS_FILTERS * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*16) x 4 x 4
            nn.Conv2d(DIS_FILTERS * 8, 1, 1, 1, 0, bias=False),
            nn.Flatten(),
            nn.Sigmoid(),
            # state size: 1 x 1 x 1
            # state size: 1
        )

    def forward(self, x):
        x = self.cons_layers(x)
        return x


class DCGANDiscriminatorNet(nn.Module):

    def __init__(self):
        super(DCGANDiscriminatorNet, self).__init__()
        self.cons_layers = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(INPUT_CHN, DIS_FILTERS, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf) x 64 x 64
            nn.Conv2d(DIS_FILTERS, DIS_FILTERS * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DIS_FILTERS * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*2) x 32 x 32
            nn.Conv2d(DIS_FILTERS * 2, DIS_FILTERS * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DIS_FILTERS * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*4) x 16 x 16
            nn.Conv2d(DIS_FILTERS * 4, DIS_FILTERS * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DIS_FILTERS * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*8) x 8 x 8
            nn.Conv2d(DIS_FILTERS * 8, DIS_FILTERS * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DIS_FILTERS * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*16) x 4 x 4
            nn.Conv2d(DIS_FILTERS * 16, 1, 4, 1, 1, bias=False),
            nn.Sigmoid(),
            # state size: 1 x 1 x 1
            nn.Flatten()
            # state size: 1
        )

    def forward(self, x):
        x = self.cons_layers(x)
        return x

class DCGANDiscriminatorNet256(nn.Module):

    def __init__(self):
        super(DCGANDiscriminatorNet256, self).__init__()
        self.cons_layers = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(INPUT_CHN, DIS_FILTERS, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size: (ndf) x 128 x 128

            nn.Conv2d(DIS_FILTERS, DIS_FILTERS * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DIS_FILTERS * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size: (ndf) x 64 x 64
            nn.Conv2d(DIS_FILTERS * 2, DIS_FILTERS * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DIS_FILTERS * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*2) x 32 x 32
            nn.Conv2d(DIS_FILTERS * 4, DIS_FILTERS * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DIS_FILTERS * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*4) x 16 x 16
            nn.Conv2d(DIS_FILTERS * 8, DIS_FILTERS * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DIS_FILTERS * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*8) x 8 x 8
            nn.Conv2d(DIS_FILTERS * 16, DIS_FILTERS * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DIS_FILTERS * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*16) x 4 x 4
            nn.Conv2d(DIS_FILTERS * 32, 1, 4, 1, 1, bias=False),
            nn.Sigmoid(),
            # state size: 1 x 1 x 1
            nn.Flatten()
            # state size: 1
        )

    def forward(self, x):
        x = self.cons_layers(x)
        return x

class SimpleDiscriminatorModel(nn.Module):

    def __init__(self):
        super(SimpleDiscriminatorModel, self).__init__()

        self.cons_layers = nn.Sequential(
            nn.Conv2d(INPUT_CHN, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(p=0.4),
            nn.Flatten(),
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cons_layers(x)
        return x


class DCGANGeneratorNet64(nn.Module):
    '''64 * 64'''

    def __init__(self):
        super(DCGANGeneratorNet64, self).__init__()
        self.cons_layers = nn.Sequential(
            nn.Unflatten(1, (LATENT_DIM, 1, 1)),  # Corrected dimension ordering

            nn.ConvTranspose2d(LATENT_DIM, GEN_FILTERS * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(GEN_FILTERS * 8),
            nn.ReLU(True),
            # state size: (ngf*16) x 4 x 4
            nn.ConvTranspose2d(GEN_FILTERS * 8, GEN_FILTERS * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FILTERS * 4),
            nn.ReLU(True),
            # state size: (ngf*8) x 8 x 8
            nn.ConvTranspose2d(GEN_FILTERS * 4, GEN_FILTERS * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FILTERS * 2),
            nn.ReLU(True),
            # state size: (ngf*4) x 16 x 16
            nn.ConvTranspose2d(GEN_FILTERS * 2, GEN_FILTERS, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FILTERS),
            nn.ReLU(True),
            # state size: (ngf*2) x 32 x 32
            nn.ConvTranspose2d(GEN_FILTERS, INPUT_CHN, 4, 2, 1, bias=False),
            # state size: (ngf) x 64 x 64
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.cons_layers(x)
        return x

class DCGANGeneratorNet(nn.Module):

    def __init__(self):
        super(DCGANGeneratorNet, self).__init__()
        self.cons_layers = nn.Sequential(
            nn.Unflatten(1, (LATENT_DIM, 1, 1)),  # Corrected dimension ordering

            nn.ConvTranspose2d(LATENT_DIM, GEN_FILTERS * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(GEN_FILTERS * 16),
            nn.ReLU(True),
            # state size: (ngf*16) x 4 x 4
            nn.ConvTranspose2d(GEN_FILTERS * 16, GEN_FILTERS * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FILTERS * 8),
            nn.ReLU(True),
            # state size: (ngf*8) x 8 x 8
            nn.ConvTranspose2d(GEN_FILTERS * 8, GEN_FILTERS * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FILTERS * 4),
            nn.ReLU(True),
            # state size: (ngf*4) x 16 x 16
            nn.ConvTranspose2d(GEN_FILTERS * 4, GEN_FILTERS * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FILTERS * 2),
            nn.ReLU(True),
            # state size: (ngf*2) x 32 x 32
            nn.ConvTranspose2d(GEN_FILTERS * 2, GEN_FILTERS, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(INPUT_CHN),
            nn.BatchNorm2d(GEN_FILTERS),
            nn.ReLU(True),
            # state size: (ngf) x 64 x 64
            nn.ConvTranspose2d(GEN_FILTERS, INPUT_CHN, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: (nc) x 128 x 128
        )

    def forward(self, x):
        x = self.cons_layers(x)
        return x

class DCGANGeneratorNet256(nn.Module):

    def __init__(self):
        super(DCGANGeneratorNet256, self).__init__()
        self.cons_layers = nn.Sequential(
            nn.Unflatten(1, (LATENT_DIM, 1, 1)),  # Corrected dimension ordering

            nn.ConvTranspose2d(LATENT_DIM, GEN_FILTERS * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(GEN_FILTERS * 32),
            nn.ReLU(True),
            # state size: (ngf*16) x 4 x 4
            nn.ConvTranspose2d(GEN_FILTERS * 32, GEN_FILTERS * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(GEN_FILTERS * 16),
            nn.ReLU(True),
            # state size: (ngf*8) x 8 x 8
            nn.ConvTranspose2d(GEN_FILTERS * 16, GEN_FILTERS * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FILTERS * 8),
            nn.ReLU(True),
            # state size: (ngf*4) x 16 x 16
            nn.ConvTranspose2d(GEN_FILTERS * 8, GEN_FILTERS * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FILTERS * 4),
            nn.ReLU(True),
            # state size: (ngf*2) x 32 x 32
            nn.ConvTranspose2d(GEN_FILTERS * 4, GEN_FILTERS * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FILTERS * 2),
            nn.ReLU(True),
            # state size: (ngf*2) x 64 x 64
            nn.ConvTranspose2d(GEN_FILTERS * 2, GEN_FILTERS, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(INPUT_CHN),
            nn.BatchNorm2d(GEN_FILTERS),
            nn.ReLU(True),
            # state size: (ngf) x 128 x 128
            nn.ConvTranspose2d(GEN_FILTERS, INPUT_CHN, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: (nc) x 256 x 256
        )

    def forward(self, x):
        x = self.cons_layers(x)
        return x



class SimpleGeneratorModel(nn.Module):

    def __init__(self, dim=LATENT_DIM):
        super(SimpleGeneratorModel, self).__init__()
        self.cons_layers = nn.Sequential(
            nn.Linear(dim, 128 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Unflatten(1, (128, 8, 8)),  # Corrected dimension ordering

            # upsample to 16 * 16
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            # upsample to 32 * 32
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(128, 3, kernel_size=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.cons_layers(x)
        return x


class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(

            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
            torch.nn.Tanh()

        )

        self.decoder = torch.nn.Sequential(

            torch.nn.Linear(2, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 128),

            torch.nn.Tanh()
        )

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, x):
        decoded = self.decoder(x)
        return decoded



class SNGANGeneratorNet(nn.Module):

    def __init__(self, *args, **kwargs):
        super(SNGANGeneratorNet, self).__init__()
        self.l1 = nn.Linear(LATENT_DIM, (BOTTOM_SIZE ** 2) * GEN_FILTERS * 16)

        self.s2 = nn.Unflatten(1, (GEN_FILTERS * 16, BOTTOM_SIZE, BOTTOM_SIZE))  # Reshape to (C, 4, 4)

        self.b3 = SNResBlockUpsample(GEN_FILTERS * 16, GEN_FILTERS * 16)
        self.b4 = SNResBlockUpsample(GEN_FILTERS * 16, GEN_FILTERS * 8)
        self.b5 = SNResBlockUpsample(GEN_FILTERS * 8, GEN_FILTERS * 8)
        self.b6 = SNResBlockUpsample(GEN_FILTERS * 8, GEN_FILTERS * 4)
        self.b7 = SNResBlockUpsample(GEN_FILTERS * 4, GEN_FILTERS * 2)
        self.b8 = SNResBlockUpsample(GEN_FILTERS * 2, GEN_FILTERS)

        self.bn9 = nn.BatchNorm2d(GEN_FILTERS)

        self.af10 = nn.LeakyReLU(0.2)

        self.conv11 = nn.Conv2d(GEN_FILTERS, INPUT_CHN, 3, 1, 1, bias=False)

    def forward(self, x):
        v = self.l1(x)
        v = self.s2(v)
        v = self.b3(v)
        v = self.b4(v)
        v = self.b5(v)
        v = self.b6(v)
        v = self.b7(v)
        v = self.b8(v)
        v = self.bn9(v)
        v = self.af10(v)
        v = self.conv11(v)
        v = F.tanh(v)
        return v


class SNGANDiscriminatorNet(nn.Module):

    def __init__(self, *args, **kwargs):
        super(SNGANDiscriminatorNet, self).__init__()
        self.b1 = SNResOptimizingBlock(INPUT_CHN, DIS_FILTERS)
        self.b2 = SNResBlockDownsample(DIS_FILTERS, DIS_FILTERS * 2)
        self.b3 = SNResBlockDownsample(DIS_FILTERS * 2, DIS_FILTERS * 4)
        self.b4 = SNResBlockDownsample(DIS_FILTERS * 4, DIS_FILTERS * 8)
        self.b5 = SNResBlockDownsample(DIS_FILTERS * 8, DIS_FILTERS * 8)
        self.b6 = SNResBlockDownsample(DIS_FILTERS * 8, DIS_FILTERS * 16)
        self.b7 = SNResBlockDownsample(DIS_FILTERS * 16, DIS_FILTERS * 16)

        self.af8 = nn.LeakyReLU(0.2)
        l9 = nn.Linear(DIS_FILTERS * 16, 1, bias=False)
        self.l9 = nn.utils.spectral_norm(l9)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        v = self.b1(x)
        v = self.b2(v)
        v = self.b3(v)
        v = self.b4(v)
        v = self.b5(v)
        v = self.b6(v)
        v = self.b7(v)
        v = self.af8(v)
        # Global average pooling
        v = self.global_avg_pool(v)
        v = v.view(v.size(0), -1)  # Flatten the output
        output = self.l9(v)
        return output


class SNResBlockUpsample(nn.Module):

    def __init__(self, in_channels, out_channels, activation = nn.LeakyReLU, hidden_channels = None, upsample=True):
        super(SNResBlockUpsample, self).__init__()
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.af1 = activation()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.af2 = activation()
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=3, padding=1)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def upsample_conv(self, x, conv):
        return conv(nn.UpsamplingNearest2d(scale_factor=2)(x))

    def residual(self, x):
        v = x
        v = self.bn1(v)
        v = self.af1(v)
        v = self.upsample_conv(v, self.conv1) if self.upsample else self.conv1(v)
        v = self.bn2(v)
        v = self.af2(v)
        v = self.conv2(v)
        return v

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class SNResBlockDownsample(nn.Module):

    def __init__(self, in_channels, out_channels, activation = nn.LeakyReLU, hidden_channels = None, kernel_size = 3, padding = 1, downsample=True):
        super(SNResBlockDownsample, self).__init__()
        self.af1 = activation()
        self.af2 = activation()
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.sp_conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding)
        self.sp_conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.sp_conv1 = nn.utils.spectral_norm(self.sp_conv1)
        self.sp_conv2 = nn.utils.spectral_norm(self.sp_conv2)

        if self.learnable_sc:
            self.sp_conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            self.sp_conv_sc = nn.utils.spectral_norm(self.sp_conv_sc)

    def residual(self, x):
        h = x
        h = self.af1(h)
        h = self.sp_conv1(h)
        h = self.af2(h)
        h = self.sp_conv2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.sp_conv_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)



def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)

class SNResOptimizingBlock(nn.Module):

    def __init__(self, in_channels, out_channels, activation = nn.LeakyReLU, kernel_size = 3, padding = 1):
        super(SNResOptimizingBlock, self).__init__()
        self.af = activation()
        self.sp_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.sp_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.sp_conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        self.sp_conv1 = nn.utils.spectral_norm(self.sp_conv1)
        self.sp_conv2 = nn.utils.spectral_norm(self.sp_conv2)
        self.sp_conv_sc = nn.utils.spectral_norm(self.sp_conv_sc)

    def residual(self, x):
        v = x
        v = self.sp_conv1(v)
        v = self.af(v)
        v = self.sp_conv2(v)
        v = _downsample(v)
        return v

    def shortcut(self, x):
        return self.sp_conv_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

