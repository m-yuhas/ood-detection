"""VGGNet Encoder and Decoder.  Based on:

K. Simonyan and A. Zisserman, ``Very Deep Convolution Networks for Large-Scale
Image Recognition,'' 3rd International Conference on Learning Representations
(ICLR), San Diego, CA, USA, May 2015.
"""

import torch


class VggnetEncBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, n_conv):
        super().__init__()
        self.conv = torch.nn.Sequential()
        self.conv.add_module(
            f'conv0',
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.conv.add_module(
            f'bn0',
            torch.nn.BatchNorm2d(out_channels),
        )
        self.conv.add_module(
            f'act0',
            torch.nn.ReLU()
        )
        for layer in range(1, n_conv):
            self.conv.add_module(f'conv{layer}', torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            self.conv.add_module(f'bn{layer}', torch.nn.BatchNorm2d(out_channels))
            self.conv.add_module(f'act{layer}', torch.nn.ReLU())
        self.conv.add_module('maxpool', torch.nn.MaxPool2d(2))
    
    def forward(self, x):
        return self.conv(x) 


class VggnetEncHead(torch.nn.Module):
    def __init__(self, in_dim, in_channels, n_latent):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels * in_dim[0] * in_dim[1], 4096)
        self.act1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.act2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(4096, n_latent)
        self.in_size = in_channels * in_dim * in_dim

    def forward(self, x):
        y = torch.reshape(x, (-1, self.in_size))
        y = self.act1(self.fc1(y))
        y = self.act2(self.fc2(y))
        y = self.fc3(y)
        return y
        #return y[:, :y.shape[-1] // 2], y[:, y.shape[-1] // 2:]


class VggnetDecBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, n_conv):
        super().__init__()
        #self.unpool = torch.nn.MaxUnpool2d(2)
        self.unpool = torch.nn.Upsample(scale_factor=2)
        self.conv = torch.nn.Sequential()
        for layer in range(n_conv - 1):
            self.conv.add_module(
                f'conv{layer}',
                torch.nn.Conv2d(in_channels, in_channels, 3, padding=1)
            )
            self.conv.add_module(
                f'bn{layer}',
                torch.nn.BatchNorm2d(in_channels)
            )
            self.conv.add_module(
                f'act{layer}',
                torch.nn.ReLU()
            )
        self.conv.add_module(
            f'convt{n_conv - 1}',
            torch.nn.ConvTranspose2d(in_channels, out_channels, 3, padding=1)
        )
        self.conv.add_module(
            f'bn{n_conv - 1}',
            torch.nn.BatchNorm2d(out_channels)
        )
        self.conv.add_module(
            f'act{n_conv - 1}',
            torch.nn.ReLU()
        )

    def forward(self, x):
        y = self.unpool(x)
        return self.conv(y)


class VggnetDecHead(torch.nn.Module):
    def __init__(self, n_latent, out_dim, out_channels):
        super().__init__()
        self.fc0 = torch.nn.Linear(n_latent, 4096)
        self.act0 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(4096, 4096)
        self.act1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(4096, out_dim[0] * out_dim[1] * out_channels)
        self.act2 = torch.nn.ReLU()
        self.out_dim = out_dim
        self.out_channels = out_channels

    def forward(self, x):
        y = self.act0(self.fc0(x))
        y = self.act1(self.fc1(y))
        y = self.act2(self.fc2(y))
        y = torch.reshape(y, (-1, self.out_channels, self.out_dim[0], self.out_dim[1]))
        return y


class VggnetEnc(torch.nn.Module):

    def __init__(self, layers: int=16, n_latent=1000, input_dim=(3, 244, 224)) -> None:
        super().__init__()

        if layers not in [11, 13, 16, 19]:
            raise('Invalid number of layers for a ResNet (valid: 11, 13, 16, 19).')

        self.block0 = VggnetEncBlock(input_dim[0], 64, 1 if layers == 11 else 2) 
        self.block1 = VggnetEncBlock(64, 128, 1 if layers == 11 else 2)
        self.block2 = VggnetEncBlock(128, 256, {11: 2, 13: 2, 16: 3, 19: 4}[layers])
        self.block3 = VggnetEncBlock(256, 512, {11: 2, 13: 2, 16: 3, 19: 4}[layers])
        self.block4 = VggnetEncBlock(512, 512, {11: 2, 13: 2, 16: 3, 19: 4}[layers])
        self.head = VggnetEncHead((input_dim[1] // 32, input_dim[2] // 32), in_channels=512, n_latent=n_latent)

    def forward(self, x):
        y = self.block0(x)
        y = self.block1(y)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.head(y)
        return y 


class VggnetDec(torch.nn.Module):

    def __init__(self, layers: int=16, n_latent=1000, output_dim=(3, 244, 224)):
        super.__init__()

        if layers not in [11, 13, 16, 19]:
            raise('Invalid number of layers for a ResNet (valid: 11, 13, 16, 19).')

        self.head = VggnetDecHead(n_latent, (output_dim[1] // 32, output_dim[2] // 32), 512)
        self.block4 = VggnetDecBlock(512, 512, {11: 2, 13: 2, 16: 3, 19: 4}[layers])
        self.block3 = VggnetEncBlock(512, 256, {11: 2, 13: 2, 16: 3, 19: 4}[layers])
        self.block2 = VggnetEncBlock(256, 128, {11: 2, 13: 2, 16: 3, 19: 4}[layers])
        self.block1 = VggnetEncBlock(128, 64, 1 if layers == 11 else 2)
        self.block0 = VggnetEncBlock(64, output_dim[0], 1 if layers == 11 else 2) 

    def forward(self, x):
        y = self.head(x)
        y = self.block4(y)
        y = self.block3(y)
        y = self.block2(y)
        y = self.block1(y)
        y = self.block0(y)
        return y 
