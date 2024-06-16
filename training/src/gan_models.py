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

from constants import *

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