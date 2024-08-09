"""Resnet Encoder and Decoder.  Based on:

K. He, X. Zhang, S. Ren, and J. Sun, ``Deep Residual Learning for Image
Recognition,'' 2016 IEEE Conference on Computer Vision and Pattern Recognition
(CVPR), Las Vegas, NV, USA, 2016, pp. 770--778, doi: 10.1109/CVPR.2016.90.
"""


from typing import Tuple
import argparse
import os


import pytorch_lightning
import torch
import torchvision


class ResnetEncBlock(torch.nn.Module):
    """Standard Resnet encoder block.  Used in Resnet 50, 101, and 152.
    
    Args:
        in_channels: number of input channels
        mid_channels: number of channels used in the middle conv layer
        out_channels: number of output channes
        stride: stride for the entire block    
    """


    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, mid_channels, 1, stride=stride, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(mid_channels)
        self.act1 = torch.nn.ReLU()
        
        self.conv2 = torch.nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(mid_channels)
        self.act2 = torch.nn.ReLU()

        self.conv3 = torch.nn.Conv2d(mid_channels, out_channels, 1, padding=0)
        self.bn3 = torch.nn.BatchNorm2d(out_channels)
        self.act3 = torch.nn.ReLU()

        self.conv_skip = torch.nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0)
        self.bn_skip = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)
        
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act2(y)

        y = self.conv3(y)
        y = self.bn3(y)
        y = self.act3(y)

        return y + self.bn_skip(self.conv_skip(x))


class ResnetEncSmallBlock(torch.nn.Module):
    """Small Resnet encoder block. Used in Resnet 18 and 34.
    
    Args:
        in_channels: number of input channels
        out_channels: number of output channes
        stride: stride for the entire block    
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.act1 = torch.nn.ReLU()
        
        self.conv2 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.act2 = torch.nn.ReLU()

        self.conv_skip = torch.nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0)
        self.bn_skip = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)
        
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act2(y)

        return y + self.bn_skip(self.conv_skip(x))


class ResnetEncMandatory(torch.nn.Module):
    """The mandatory block of any Resnet."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn = torch.nn.BatchNorm2d(64)
        self.act = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        return self.pool(y)

class ResnetEncHead(torch.nn.Module):
    """A resnet head leading from a (c, h, w) tensor to a 1D tensor of
    perdictions.

    Args:
        in_dim: input dimensions (h, w)
        in_channels: number of input channels
        n_latent: output tensor size
    """

    def __init__(self, in_dim: Tuple[int, int], in_channels: int, n_latent: int) -> None:
        super().__init__()
        self.pool = torch.nn.AvgPool2d(in_dim)
        self.fc = torch.nn.Linear(in_channels, n_latent)
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)
        y = torch.reshape(y, (-1, self.in_channels))
        y = self.fc(y)
        return y


class ResnetDecBlock(torch.nn.Module):
    """Resnet decoder block for Resnet 50, 101, and 152.  Because a Resnet
    decoder is not described in the original paper, here, convolutional layers
    are replaced with transpose convolutions.
    
    Args:
        in_channels: number of input channels
        mid_channels: number of channels in the middle convolution
        out_channels: number of output channels
        stride: stride (upsample amount) for the entire block
    """

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(in_channels, mid_channels, 1, stride=stride, padding=0, output_padding=stride-1)
        self.bn1 = torch.nn.BatchNorm2d(mid_channels)
        self.act1 = torch.nn.ReLU()

        self.conv2 = torch.nn.ConvTranspose2d(mid_channels, mid_channels, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(mid_channels)
        self.act2 = torch.nn.ReLU()

        self.conv3 = torch.nn.ConvTranspose2d(mid_channels, out_channels, 1, padding=0)
        self.bn3 = torch.nn.BatchNorm2d(out_channels)
        self.act3 = torch.nn.ReLU()

        self.conv_skip = torch.nn.ConvTranspose2d(in_channels, out_channels, 1, stride=stride, padding=0, output_padding=stride-1)
        self.bn_skip = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act2(y)

        y = self.conv3(y)
        y = self.bn3(y)
        y = self.act3(y)

        return y + self.bn_skip(self.conv_skip(x))


class ResnetDecSmallBlock(torch.nn.Module):
    """Small Resnet decoder block for Resnet 18 and 34.  Because a Resnet
    decoder is not described in the original paper, here, convolutional layers
    are replaced with transpose convolutions.
    
    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        stride: stride (upsample amount) for the entire block
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(in_channels, out_channels, 3, stride=stride, padding=1, output_padding=stride-1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.act1 = torch.nn.ReLU()

        self.conv2 = torch.nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.act2 = torch.nn.ReLU()

        self.conv_skip = torch.nn.ConvTranspose2d(in_channels, out_channels, 1, stride=stride, padding=0, output_padding=stride-1)
        self.bn_skip = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act2(y)

        return y + self.bn_skip(self.conv_skip(x))


class ResnetDecMandatory(torch.nn.Module):
    """The mandatory portion of a Resnet decoder.  Convolutions are replaced
    with their corresponding transpose convolutions and the max pool is
    replaced with a transposed convolution with 3x3 learnable filters grouped
    by channel.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(64, 3, 7, stride=2, padding=3, output_padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.act = torch.nn.ReLU()
        self.pool = torch.nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1, groups=64)
        self.pool_bn = torch.nn.BatchNorm2d(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool_bn(self.pool(x))
        y = self.conv(y)
        y = self.bn(y)
        return self.act(y)


class ResnetDecHead(torch.nn.Module):
    """The input block to a Resnet decoder.  The pooling layer is replaced
    with a transpose convolution with per channel learnable filters.
    
    Args:
        n_latent: input tensor length
        out_dim: output dimensions (h, w)
        out_channels: number of output channels
    """

    def __init__(self, n_latent: int, out_dim: Tuple[int, int], out_channels: int) -> None:
        super().__init__()
        self.pool = torch.nn.ConvTranspose2d(out_channels, out_channels, out_dim, stride=1, groups=out_channels)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.fc = torch.nn.Linear(n_latent, out_channels)
        self.out_dim = out_dim,
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        y = torch.reshape(y, (-1, self.out_channels, 1, 1))
        return self.bn(self.pool(y))


class Resnet18Enc(torch.nn.Module):
    """Resnet 18 encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.block0_0 = ResnetEncMandatory()

        self.block1_0 = ResnetEncSmallBlock(64, 64, 1)
        self.block1_1 = ResnetEncSmallBlock(64, 64, 1)

        self.block2_0 = ResnetEncSmallBlock(64, 128, 2)
        self.block2_1 = ResnetEncSmallBlock(128, 128, 1)

        self.block3_0 = ResnetEncSmallBlock(128, 256, 2)
        self.block3_1 = ResnetEncSmallBlock(256, 256, 1)

        self.block4_0 = ResnetEncSmallBlock(256, 512, 2)
        self.block4_1 = ResnetEncSmallBlock(512, 512, 1)

        self.head = ResnetEncHead(7, 512, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block0_0(x)

        y = self.block1_0(y)
        y = self.block1_1(y)

        y = self.block2_0(y)
        y = self.block2_1(y)

        y = self.block3_0(y)
        y = self.block3_1(y)

        y = self.block4_0(y)
        y = self.block4_1(y)

        y = self.head(y)
        return y


class Resnet34Enc(torch.nn.Module):
    """Resnet 34 encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.block0_0 = ResnetEncMandatory()

        self.block1_0 = ResnetEncSmallBlock(64, 64, 1)
        self.block1_1 = ResnetEncSmallBlock(64, 64, 1)
        self.block1_2 = ResnetEncSmallBlock(64, 64, 1)

        self.block2_0 = ResnetEncSmallBlock(64, 128, 2)
        self.block2_1 = ResnetEncSmallBlock(128, 128, 1)
        self.block2_2 = ResnetEncSmallBlock(128, 128, 1)
        self.block2_3 = ResnetEncSmallBlock(128, 128, 1)

        self.block3_0 = ResnetEncSmallBlock(128, 256, 2)
        self.block3_1 = ResnetEncSmallBlock(256, 256, 1)
        self.block3_2 = ResnetEncSmallBlock(256, 256, 1)
        self.block3_3 = ResnetEncSmallBlock(256, 256, 1)
        self.block3_4 = ResnetEncSmallBlock(256, 256, 1)
        self.block3_5 = ResnetEncSmallBlock(256, 256, 1)

        self.block4_0 = ResnetEncSmallBlock(256, 512, 2)
        self.block4_1 = ResnetEncSmallBlock(512, 512, 1)
        self.block4_2 = ResnetEncSmallBlock(512, 512, 1)

        self.head = ResnetEncHead(7, 512, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block0_0(x)

        y = self.block1_0(y)
        y = self.block1_1(y)
        y = self.block1_2(y)

        y = self.block2_0(y)
        y = self.block2_1(y)
        y = self.block2_2(y)
        y = self.block2_3(y)

        y = self.block3_0(y)
        y = self.block3_1(y)
        y = self.block3_2(y)
        y = self.block3_3(y)
        y = self.block3_4(y)
        y = self.block3_5(y)

        y = self.block4_0(y)
        y = self.block4_1(y)
        y = self.block4_2(y)

        y = self.head(y)
        return y


class Resnet18Dec(torch.nn.Module):
    """Resnet 18 decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.head = ResnetDecHead(500, 7, 512)

        self.block4_1 = ResnetDecSmallBlock(512, 512, 1)
        self.block4_0 = ResnetDecSmallBlock(512, 256, 2)

        self.block3_1 = ResnetDecSmallBlock(256, 256, 1)
        self.block3_0 = ResnetDecSmallBlock(256, 128, 2)

        self.block2_1 = ResnetDecSmallBlock(128, 128, 1)
        self.block2_0 = ResnetDecSmallBlock(128, 64, 2)

        self.block1_1 = ResnetDecSmallBlock(64, 64, 1)
        self.block1_0 = ResnetDecSmallBlock(64, 64, 1)

        self.block0_0 = ResnetDecMandatory()

    def forward(self, x):
        y = self.head(x)

        y = self.block4_1(y)
        y = self.block4_0(y)

        y = self.block3_1(y)
        y = self.block3_0(y)

        y = self.block2_1(y)
        y = self.block2_0(y)

        y = self.block1_1(y)
        y = self.block1_0(y)

        return self.block0_0(y)
    

class Resnet34Dec(torch.nn.Module):
    """Resnet 34 decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.head = ResnetDecHead(500, 7, 512)

        self.block4_2 = ResnetDecSmallBlock(512, 512, 1)
        self.block4_1 = ResnetDecSmallBlock(512, 512, 1)
        self.block4_0 = ResnetDecSmallBlock(512, 256, 2)

        self.block3_5 = ResnetDecSmallBlock(256, 256, 1)
        self.block3_4 = ResnetDecSmallBlock(256, 256, 1)
        self.block3_3 = ResnetDecSmallBlock(256, 256, 1)
        self.block3_2 = ResnetDecSmallBlock(256, 256, 1)
        self.block3_1 = ResnetDecSmallBlock(256, 256, 1)
        self.block3_0 = ResnetDecSmallBlock(256, 128, 2)

        self.block2_3 = ResnetDecSmallBlock(128, 128, 1)
        self.block2_2 = ResnetDecSmallBlock(128, 128, 1)
        self.block2_1 = ResnetDecSmallBlock(128, 128, 1)
        self.block2_0 = ResnetDecSmallBlock(128, 64, 2)

        self.block1_2 = ResnetDecSmallBlock(64, 64, 1)
        self.block1_1 = ResnetDecSmallBlock(64, 64, 1)
        self.block1_0 = ResnetDecSmallBlock(64, 64, 1)

        self.block0_0 = ResnetDecMandatory()

    def forward(self, x):
        y = self.head(x)

        y = self.block4_2(y)
        y = self.block4_1(y)
        y = self.block4_0(y)

        y = self.block3_5(y)
        y = self.block3_4(y)
        y = self.block3_3(y)
        y = self.block3_2(y)
        y = self.block3_1(y)
        y = self.block3_0(y)

        y = self.block2_3(y)
        y = self.block2_2(y)
        y = self.block2_1(y)
        y = self.block2_0(y)

        y = self.block1_2(y)
        y = self.block1_1(y)
        y = self.block1_0(y)

        return self.block0_0(y)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a Resnet VAE')
    parser.add_argument(
        '--train_dataset',
        help='Path to folder of training images'
    )
    parser.add_argument(
        '--val_dataset',
        help='Path to folder of validation images'
    )
    parser.add_argument(
        '--batch',
        help='Batch size'
    )
    parser.add_argument(
        '--epochs',
        help='Number of epochs to train'
    )
    parser.add_argument(
        '--name',
        help='Name of output model'
    )
    args = parser.parse_args()
    model = Resnet18Vae()
    data = OodDataModule(args.train_dataset, args.val_dataset, None, None, batch_size=int(args.batch)) 
    trainer = pytorch_lightning.Trainer(
        accelerator='gpu',
        devices=1,
        deterministic=True,
        min_epochs=50,
        max_epochs=int(args.epochs),
        log_every_n_steps=1,
        logger=pytorch_lightning.loggers.CSVLogger("logs", name=args.name),
    )
    trainer.fit(model, datamodule=data)
    torch.save(model, args.name + '.pt')