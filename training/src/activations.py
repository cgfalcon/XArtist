import torch.nn as nn
import torch
import torch.nn.functional as F


class AFFactory():
    "An activation function factory"

    def __init__(self):
        self.afs = {
            'FReLU': FReLU,
            'ELU': nn.ELU,
            'ReLU': nn.ReLU,
            'LeakyADA': LeakyADA,
            'ShiftedSincUnit': ShiftedSincUnit,
            'ShiftedQuadraticUnit': ShiftedQuadraticUnit,
            'GCU': GCU,
            'ADA': ADA,
            'Swish': Swish,
            'Sigmoid': nn.Sigmoid
        }

    def get_activation(self, af_name, af_params):
        if af_name not in self.afs:
            raise KeyError('Unknown activation function: {}'.format(af_name))

        return self.afs[af_name](**af_params)

class FReLU(nn.Module):

    def __init__(self, frelu_init = 0.2, inplace = False):
        super(FReLU, self).__init__()
        if frelu_init == 0:
            self.b = nn.Parameter(torch.zeros(1))
        else:
            self.b = nn.Parameter(torch.ones(1) * frelu_init)
        self.inplace = inplace

    def forward(self, x):
        return F.relu(x, inplace=self.inplace) + self.b

    def extra_repr(self) -> str:
        repr = f'inplace={"True" if self.inplace else ""}, bias=({self.b.shape})'
        return repr


class LeakyADA(nn.Module):
    def __init__(self, alpha=0.5, leak=0.01):
        super(LeakyADA, self).__init__()
        self.alpha = alpha
        self.leak = leak

    def forward(self, x):
        return self.leak * torch.min(x, torch.tensor(0.0)) + torch.max(x, torch.tensor(0.0)) * torch.exp(
            -x * self.alpha)


    def extra_repr(self) -> str:
        repr = f'leak={self.leak}, alpha={self.alpha}'
        return repr

class ShiftedSincUnit(nn.Module):

    def __init__(self):
        super(ShiftedSincUnit, self).__init__()

    def sinc(self, z):
        return torch.where(z == 0, 1, torch.sin(z) / z)

    def forward(self, x):
        return torch.pi * self.sinc(x - torch.pi)

class GCU(nn.Module):

    def __init__(self):
        super(GCU, self).__init__()

    def forward(self, x):
        return x * torch.cos(x)

class ADA(nn.Module):

    def __init__(self, alpha=0.5):
        super(ADA, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.maximum(x, torch.tensor(0.)) * torch.exp(-x * self.alpha)

class Swish(nn.Module):

    def __init__(self, alpha=1):
        super(Swish, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return x * torch.sigmoid(self.alpha * x)

class ShiftedQuadraticUnit(nn.Module):

    def __init__(self):
        super(ShiftedQuadraticUnit, self).__init__()

    def forward(self, x):
        return x ** 2 + x