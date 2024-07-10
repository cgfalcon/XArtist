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

from src.activations import FReLU, AFFactory

from src.configs import *
from src.layers import *

INPUT_CHN = train_configs['INPUT_CHN']
DIS_FILTERS = train_configs['DIS_FILTERS']
LATENT_DIM = train_configs['LATENT_DIM']
GEN_FILTERS = train_configs['GEN_FILTERS']
BOTTOM_SIZE = train_configs['BOTTOM_SIZE']

class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()

        self.gan = nn.Sequential(
            generator,
            discriminator
        )


    def forward(self, x):
        return self.gan(x)


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
            # state size: 1 x 1 x 1
            # state size: 1
        )

    def forward(self, x):
        x = self.cons_layers(x)
        return x


class SNDCGANDiscriminatorNet64(nn.Module):
    '''64 * 64'''

    def __init__(self):
        super(SNDCGANDiscriminatorNet64, self).__init__()
        self.cons_layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(INPUT_CHN, DIS_FILTERS, 4, 2, 1, bias=False)),
            # nn.BatchNorm2d(DIS_FILTERS),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*2) x 32 x 32
            nn.utils.spectral_norm(nn.Conv2d(DIS_FILTERS, DIS_FILTERS * 2, 4, 2, 1, bias=False)),
            # nn.BatchNorm2d(DIS_FILTERS * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*4) x 16 x 16
            nn.utils.spectral_norm(nn.Conv2d(DIS_FILTERS * 2, DIS_FILTERS * 4, 4, 2, 1, bias=False)),
            # nn.BatchNorm2d(DIS_FILTERS * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*8) x 8 x 8
            nn.utils.spectral_norm(nn.Conv2d(DIS_FILTERS * 4, DIS_FILTERS * 8, 8, 2, 1, bias=False)),
            # nn.BatchNorm2d(DIS_FILTERS * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*16) x 4 x 4
            nn.utils.spectral_norm(nn.Conv2d(DIS_FILTERS * 8, 1, 1, 1, 0, bias=False)),
            nn.Flatten(),
            # state size: 1 x 1 x 1
            # state size: 1
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
            # state size: 1 x 1 x 1
            nn.Flatten()
            # state size: 1
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
            FReLU(inplace=True),
            # state size: (ngf*16) x 4 x 4
            nn.ConvTranspose2d(GEN_FILTERS * 16, GEN_FILTERS * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FILTERS * 8),
            FReLU(inplace=True),
            # state size: (ngf*8) x 8 x 8
            nn.ConvTranspose2d(GEN_FILTERS * 8, GEN_FILTERS * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FILTERS * 4),
            FReLU(inplace=True),
            # state size: (ngf*4) x 16 x 16
            nn.ConvTranspose2d(GEN_FILTERS * 4, GEN_FILTERS * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FILTERS * 2),
            FReLU(inplace=True),
            # state size: (ngf*2) x 32 x 32
            nn.ConvTranspose2d(GEN_FILTERS * 2, GEN_FILTERS, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FILTERS),
            FReLU(inplace=True),
            # state size: (ngf) x 64 x 64
            nn.ConvTranspose2d(GEN_FILTERS, INPUT_CHN, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: (nc) x 128 x 128
        )

    def forward(self, x):
        x = self.cons_layers(x)
        return x


class SNDCGANDiscriminatorNet(nn.Module):
    '''Spectral Normalized DCGAN Discriminator'''

    def __init__(self):
        super(SNDCGANDiscriminatorNet, self).__init__()
        self.cons_layers = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.utils.spectral_norm(nn.Conv2d(INPUT_CHN, DIS_FILTERS, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf) x 64 x 64
            nn.utils.spectral_norm(nn.Conv2d(DIS_FILTERS, DIS_FILTERS * 2, 4, 2, 1)),
            # nn.BatchNorm2d(DIS_FILTERS * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*2) x 32 x 32
            nn.utils.spectral_norm(nn.Conv2d(DIS_FILTERS * 2, DIS_FILTERS * 4, 4, 2, 1)),
            # nn.BatchNorm2d(DIS_FILTERS * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*4) x 16 x 16
            nn.utils.spectral_norm(nn.Conv2d(DIS_FILTERS * 4, DIS_FILTERS * 8, 4, 2, 1)),
            # nn.BatchNorm2d(DIS_FILTERS * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*8) x 8 x 8
            nn.utils.spectral_norm(nn.Conv2d(DIS_FILTERS * 8, DIS_FILTERS * 16, 4, 2, 1)),
            # nn.BatchNorm2d(DIS_FILTERS * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*16) x 4 x 4
            nn.utils.spectral_norm(nn.Conv2d(DIS_FILTERS * 16, 1, 4, 1, 0)),
            # state size: 1 x 1 x 1
            nn.Flatten()
            # state size: 1
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
            FReLU(0.2, inplace=True),
            # state size: (ngf*16) x 4 x 4
            nn.ConvTranspose2d(GEN_FILTERS * 32, GEN_FILTERS * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(GEN_FILTERS * 16),
            FReLU(0.2, inplace=True),
            # state size: (ngf*8) x 8 x 8
            nn.ConvTranspose2d(GEN_FILTERS * 16, GEN_FILTERS * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FILTERS * 8),
            FReLU(0.2, inplace=True),
            # state size: (ngf*4) x 16 x 16
            nn.ConvTranspose2d(GEN_FILTERS * 8, GEN_FILTERS * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FILTERS * 4),
            FReLU(0.2, inplace=True),
            # state size: (ngf*2) x 32 x 32
            nn.ConvTranspose2d(GEN_FILTERS * 4, GEN_FILTERS * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FILTERS * 2),
            FReLU(0.2, inplace=True),
            # state size: (ngf*2) x 64 x 64
            nn.ConvTranspose2d(GEN_FILTERS * 2, GEN_FILTERS, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(INPUT_CHN),
            nn.BatchNorm2d(GEN_FILTERS),
            FReLU(0.2, inplace=True),
            # state size: (ngf) x 128 x 128
            nn.ConvTranspose2d(GEN_FILTERS, INPUT_CHN, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: (nc) x 256 x 256
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
            # state size: 1 x 1 x 1
            nn.Flatten()
            # state size: 1
        )

    def forward(self, x):
        x = self.cons_layers(x)
        return x

class SNDCGANDiscriminatorNet256(nn.Module):

    def __init__(self):
        super(SNDCGANDiscriminatorNet256, self).__init__()
        self.cons_layers = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.utils.spectral_norm(nn.Conv2d(INPUT_CHN, DIS_FILTERS, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # state size: (ndf) x 128 x 128

            nn.utils.spectral_norm(nn.Conv2d(DIS_FILTERS, DIS_FILTERS * 2, 4, 2, 1)),
            # nn.BatchNorm2d(DIS_FILTERS * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size: (ndf) x 64 x 64
            nn.utils.spectral_norm(nn.Conv2d(DIS_FILTERS * 2, DIS_FILTERS * 4, 4, 2, 1)),
            # nn.BatchNorm2d(DIS_FILTERS * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*2) x 32 x 32
            nn.utils.spectral_norm(nn.Conv2d(DIS_FILTERS * 4, DIS_FILTERS * 8, 4, 2, 1)),
            # nn.BatchNorm2d(DIS_FILTERS * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*4) x 16 x 16
            nn.utils.spectral_norm(nn.Conv2d(DIS_FILTERS * 8, DIS_FILTERS * 16, 4, 2, 1)),
            # nn.BatchNorm2d(DIS_FILTERS * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*8) x 8 x 8
            nn.utils.spectral_norm(nn.Conv2d(DIS_FILTERS * 16, DIS_FILTERS * 32, 4, 2, 1)),
            # nn.BatchNorm2d(DIS_FILTERS * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*16) x 4 x 4
            nn.utils.spectral_norm(nn.Conv2d(DIS_FILTERS * 32, 1, 4, 1, 1)),
            # nn.Sigmoid(),
            # state size: 1 x 1 x 1
            nn.Flatten()
            # state size: 1
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

        self.af10 = nn.ReLU(inplace=True)

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
        self.b7 = SNResBlockDownsample(DIS_FILTERS * 16, DIS_FILTERS * 16, downsample=False)

        self.af8 = nn.ReLU(0.2)
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

class SNGANGeneratorNet128(nn.Module):

    def __init__(self, *args, **kwargs):
        super(SNGANGeneratorNet128, self).__init__()
        self.l1 = nn.Linear(LATENT_DIM, (BOTTOM_SIZE ** 2) * GEN_FILTERS * 16)

        self.s2 = nn.Unflatten(1, (GEN_FILTERS * 16, BOTTOM_SIZE, BOTTOM_SIZE))  # Reshape to (C, 4, 4)

        self.b3 = SNResBlockUpsample(GEN_FILTERS * 16, GEN_FILTERS * 16)
        self.b4 = SNResBlockUpsample(GEN_FILTERS * 16, GEN_FILTERS * 8)
        self.b5 = SNResBlockUpsample(GEN_FILTERS * 8, GEN_FILTERS * 8)
        self.b6 = SNResBlockUpsample(GEN_FILTERS * 8, GEN_FILTERS * 4)
        self.b7 = SNResBlockUpsample(GEN_FILTERS * 4, GEN_FILTERS * 2)

        self.bn8 = nn.BatchNorm2d(GEN_FILTERS * 2)

        self.af9 = FReLU(0.2)

        self.conv10 = nn.Conv2d(GEN_FILTERS * 2, INPUT_CHN, 3, 1, 1)

    def forward(self, x):
        v = self.l1(x)
        v = self.s2(v)
        v = self.b3(v)
        v = self.b4(v)
        v = self.b5(v)
        v = self.b6(v)
        v = self.b7(v)
        v = self.bn8(v)
        v = self.af9(v)
        v = self.conv10(v)
        v = F.tanh(v)
        return v


class SNGANDiscriminatorNet128(nn.Module):

    def __init__(self, *args, **kwargs):
        super(SNGANDiscriminatorNet128, self).__init__()
        self.b1 = SNResOptimizingBlock(INPUT_CHN, DIS_FILTERS)
        self.b2 = SNResBlockDownsample(DIS_FILTERS, DIS_FILTERS * 2)
        self.b3 = SNResBlockDownsample(DIS_FILTERS * 2, DIS_FILTERS * 4)
        self.b4 = SNResBlockDownsample(DIS_FILTERS * 4, DIS_FILTERS * 8)
        self.b5 = SNResBlockDownsample(DIS_FILTERS * 8, DIS_FILTERS * 8)
        self.b6 = SNResBlockDownsample(DIS_FILTERS * 8, DIS_FILTERS * 16)

        self.af7 = FReLU(0.2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        l8 = nn.Linear(DIS_FILTERS * 16, 1, bias=False)
        self.l8 = nn.utils.spectral_norm(l8)

    def forward(self, x):
        v = self.b1(x)
        v = self.b2(v)
        v = self.b3(v)
        v = self.b4(v)
        v = self.b5(v)
        v = self.b6(v)
        v = self.af7(v)
        # Global average pooling
        v = self.global_avg_pool(v)
        v = v.view(v.size(0), -1)  # Flatten the output
        output = self.l8(v)
        return output


class ResGANGeneratorNet(nn.Module):

    def __init__(self, *args, **kwargs):
        super(ResGANGeneratorNet, self).__init__()
        af = nn.LeakyReLU
        self.l1 = nn.Linear(LATENT_DIM, (BOTTOM_SIZE ** 2) * GEN_FILTERS * 16)

        self.s2 = nn.Unflatten(1, (GEN_FILTERS * 16, BOTTOM_SIZE, BOTTOM_SIZE))  # Reshape to (C, 4, 4)

        self.b3 = ResBlockUpsample(GEN_FILTERS * 16, GEN_FILTERS * 16, activation=af)
        self.b4 = ResBlockUpsample(GEN_FILTERS * 16, GEN_FILTERS * 8, activation=af)
        self.b5 = ResBlockUpsample(GEN_FILTERS * 8, GEN_FILTERS * 8, activation=af)
        self.b6 = ResBlockUpsample(GEN_FILTERS * 8, GEN_FILTERS * 4, activation=af)
        self.b7 = ResBlockUpsample(GEN_FILTERS * 4, GEN_FILTERS * 2, activation=af)
        self.b8 = ResBlockUpsample(GEN_FILTERS * 2, GEN_FILTERS, activation=af)

        self.bn9 = nn.BatchNorm2d(GEN_FILTERS)

        self.af10 = af(0.2)

        self.conv11 = nn.Conv2d(GEN_FILTERS, INPUT_CHN, 3, 1, 1)

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
        v = nn.Tanh()(v)
        return v


class ResGANDiscriminatorNet(nn.Module):

    def __init__(self, *args, **kwargs):
        super(ResGANDiscriminatorNet, self).__init__()
        af = nn.LeakyReLU
        self.b1 = ResOptimizingBlock(INPUT_CHN, DIS_FILTERS, activation=af)
        self.b2 = ResBlockDownsample(DIS_FILTERS, DIS_FILTERS * 2, activation=af)
        self.b3 = ResBlockDownsample(DIS_FILTERS * 2, DIS_FILTERS * 4, activation=af)
        self.b4 = ResBlockDownsample(DIS_FILTERS * 4, DIS_FILTERS * 8, activation=af)
        self.b5 = ResBlockDownsample(DIS_FILTERS * 8, DIS_FILTERS * 8, activation=af)
        self.b6 = ResBlockDownsample(DIS_FILTERS * 8, DIS_FILTERS * 16, activation=af)
        self.b7 = ResBlockDownsample(DIS_FILTERS * 16, DIS_FILTERS * 16, activation=af)

        self.af8 = af(0.2)
        self.l9 = nn.Linear(DIS_FILTERS * 16, 1, bias=False)

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
        v = v.sum(2).sum(2)
        output = self.l9(v)
        return output

class ResGANGeneratorNet128(nn.Module):

    def __init__(self, *args, **kwargs):
        super(ResGANGeneratorNet128, self).__init__()
        af = nn.LeakyReLU
        self.l1 = nn.Linear(LATENT_DIM, (BOTTOM_SIZE ** 2) * GEN_FILTERS * 16)

        self.s2 = nn.Unflatten(1, (GEN_FILTERS * 16, BOTTOM_SIZE, BOTTOM_SIZE))  # Reshape to (C, 4, 4)

        self.b3 = ResBlockUpsample(GEN_FILTERS * 16, GEN_FILTERS * 16, activation=af)
        self.b4 = ResBlockUpsample(GEN_FILTERS * 16, GEN_FILTERS * 8, activation=af)
        self.b5 = ResBlockUpsample(GEN_FILTERS * 8, GEN_FILTERS * 8, activation=af)
        self.b6 = ResBlockUpsample(GEN_FILTERS * 8, GEN_FILTERS * 4, activation=af)
        self.b7 = ResBlockUpsample(GEN_FILTERS * 4, GEN_FILTERS * 2, activation=af)

        self.bn8 = nn.BatchNorm2d(GEN_FILTERS * 2)

        self.af9 = af(0.2)

        self.conv10 = nn.Conv2d(GEN_FILTERS * 2, INPUT_CHN, 3, 1, 1, bias=False)

    def forward(self, x):
        v = self.l1(x)
        v = self.s2(v)
        v = self.b3(v)
        v = self.b4(v)
        v = self.b5(v)
        v = self.b6(v)
        v = self.b7(v)
        v = self.bn8(v)
        v = self.af9(v)
        v = self.conv10(v)
        v = nn.Tanh()(v)
        return v


class ResGANDiscriminatorNet128(nn.Module):

    def __init__(self, *args, **kwargs):
        super(ResGANDiscriminatorNet128, self).__init__()
        af = nn.LeakyReLU
        self.b1 = ResOptimizingBlock(INPUT_CHN, DIS_FILTERS, activation=af)
        self.b2 = ResBlockDownsample(DIS_FILTERS, DIS_FILTERS * 2, activation=af)
        self.b3 = ResBlockDownsample(DIS_FILTERS * 2, DIS_FILTERS * 4, activation=af)
        self.b4 = ResBlockDownsample(DIS_FILTERS * 4, DIS_FILTERS * 8, activation=af)
        self.b5 = ResBlockDownsample(DIS_FILTERS * 8, DIS_FILTERS * 8, activation=af)
        self.b6 = ResBlockDownsample(DIS_FILTERS * 8, DIS_FILTERS * 16, activation=af)

        self.af7 = af(0.2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.l8 = nn.Linear(DIS_FILTERS * 16, 1, bias=False)

    def forward(self, x):
        v = self.b1(x)
        v = self.b2(v)
        v = self.b3(v)
        v = self.b4(v)
        v = self.b5(v)
        v = self.b6(v)
        v = self.af7(v)
        # Global average pooling
        v = self.global_avg_pool(v)
        v = v.view(v.size(0), -1)  # Flatten the output
        output = self.l8(v)
        return output