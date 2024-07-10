from torch import nn
from src.activations import *

class SNResBlockUpsample(nn.Module):

    def __init__(self, in_channels, out_channels, activation = FReLU, hidden_channels = None, upsample=True):
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

    def __init__(self, in_channels, out_channels, activation = FReLU, hidden_channels = None, kernel_size = 3, padding = 1, downsample=True):
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

    def __init__(self, in_channels, out_channels, activation = FReLU, kernel_size = 3, padding = 1):
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


class ResBlockUpsample(nn.Module):

    def __init__(self, in_channels, out_channels, activation, hidden_channels = None, upsample=True):
        super(ResBlockUpsample, self).__init__()
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


class ResBlockDownsample(nn.Module):

    def __init__(self, in_channels, out_channels, activation, hidden_channels = None, kernel_size = 3, padding = 1, downsample=True):
        super(ResBlockDownsample, self).__init__()
        self.af1 = activation()
        self.af2 = activation()
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.sp_conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding)
        self.sp_conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, padding=padding)
        # self.sp_conv1 = nn.utils.spectral_norm(self.sp_conv1)
        # self.sp_conv2 = nn.utils.spectral_norm(self.sp_conv2)

        if self.learnable_sc:
            self.sp_conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            # self.sp_conv_sc = nn.utils.spectral_norm(self.sp_conv_sc)

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

class ResOptimizingBlock(nn.Module):

    def __init__(self, in_channels, out_channels, activation = nn.LeakyReLU, kernel_size = 3, padding = 1):
        super(ResOptimizingBlock, self).__init__()
        self.af = activation()
        self.sp_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.sp_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.sp_conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        # self.sp_conv1 = nn.utils.spectral_norm(self.sp_conv1)
        # self.sp_conv2 = nn.utils.spectral_norm(self.sp_conv2)
        # self.sp_conv_sc = nn.utils.spectral_norm(self.sp_conv_sc)

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