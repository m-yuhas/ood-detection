"""Mobilenet Encoder and Decoder.  Based on:

A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M.
Andreetto, and Hartwig Adam, ``MobileNets: Efficient Convolutional Neural
Networks for Mobile Vision Applications,'' 2017, arXiv:1704.04861.
"""


from typing import Tuple
import argparse
import os


import pytorch_lightning
import torch
import torchvision


class ConvEncBlock(torch.nn.Module):
    """MobileNet Conv block.
    
    Args:
        in_channels: number of input channels
        out_channels: number of output channes
        stride: stride for the entire block
        kernel_size: kernel size 
    """


    def __init__(self, in_channels: int, out_channels: int, stride: int, kernel_size: int=3) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.act = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseConvEncBlock(torch.nn.Module):
    """Small Resnet encoder block. Used in Resnet 18 and 34.
    
    Args:
        in_channels: number of input channels
        out_channels: number of output channes
        stride: stride for the entire block
        kernel_size: kernel size (for depth-wise conv)  
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.dwconv = torch.nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=kernel_size // 2, groups=in_channels)
        self.dwbn = torch.nn.BatchNorm2d(in_channels)
        self.dwact = torch.nn.ReLU()
        
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 1)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.act = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dwconv(x)
        y = self.dwbn(y)
        y = self.dwact(y)
        
        y = self.conv(y)
        y = self.bn(y)
        y = self.act(y)

        return y

class MobilenetEncHead(torch.nn.Module):
    """A MobileNet head leading from a (c, h, w) tensor to a 1D tensor of
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


class ConvDecBlock(torch.nn.Module):
    """Upsample and use regular convolutions to allow quantization.
    
    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        stride: stride (upsample amount) for the entire block
        kernel_size: convolutional kernel size
    """

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.upsample = torch.nn.Upsample(stride)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.act = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(self.upsample(x))))


class DepthwiseConvDecBlock(torch.nn.Module):
    """Upsample and use regular convolutions to allow quantization.
    
    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        stride: stride (upsample amount) for the entire block
        kernel_size: convolutional kernel size (depthwise conv only)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 1)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.act = torch.nn.ReLU()

        self.upsample = torch.nn.Upsample(stride)
        self.dwconv = torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2, groups=in_channels)
        self.dwbn = torch.nn.BatchNorm2d(out_channels)
        self.dwact = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)

        y = self.upsample(y)
        y = self.dwconv(y)
        y = self.dwbn(y)
        y = self.dwact(y)

        return y

class MobilenetDecHead(torch.nn.Module):
    """The input block to a Resnet decoder.  The pooling layer is replaced
    with an upsample.
    
    Args:
        n_latent: input tensor length
        out_dim: output dimensions (h, w)
        out_channels: number of output channels
    """

    def __init__(self, n_latent: int, out_dim: Tuple[int, int], out_channels: int) -> None:
        super().__init__()
        self.pool = torch.nn.Upsample(out_dim)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.fc = torch.nn.Linear(n_latent, out_channels)
        self.out_dim = out_dim,
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        y = torch.reshape(y, (-1, self.out_channels, 1, 1))
        return self.bn(self.pool(y))


class MobileNetEnc(torch.nn.Module):
    """Resnet 18 encoder."""

    def __init__(self, alpha=1, n_latent=1000, input_shape=(3,224,224)) -> None:
        super().__init__()
        self.block0 = ConvEncBlock(input_shape[0], int(32 * alpha), stride=2)

        self.block1 = torch.nn.ModuleList([
            DepthwiseConvEncBlock(int(alpha * 32), int(alpha * 32), stride=1),
            ConvEncBlock(int(alpha * 32), int(alpha * 128), stride=1),
            DepthwiseConvEncBlock(int(alpha * 64), int(alpha * 64), stride=2),
        ])

        self.block2 = torch.nn.ModuleList([
            ConvEncBlock(int(alpha * 64), int(alpha * 128), stride=1, kernel_size=1),
            DepthwiseConvEncBlock(int(alpha * 128), int(alpha * 128), stride=1),
            ConvEncBlock(int(alpha * 128), int(alpha * 128), stride=1, kernel_size=1),
            DepthwiseConvDecBlock(int(alpha * 128), int(alpha * 128), stride=2),
        ])

        self.block3 = torch.nn.ModuleList([
            ConvEncBlock(int(alpha * 128), int(alpha * 256), stride=1, kernel_size=1),
            DepthwiseConvEncBlock(int(alpha * 256), int(alpha * 256), stride=1),
            ConvEncBlock(int(alpha *256), int(alpha * 256), stride=1, kernel_size=1),
            DepthwiseConvEncBlock(int(alpha * 256), int(alpha * 256), stride=2),
        ])

        self.block4 = torch.nn.ModuleList([
            ConvEncBlock(int(alpha * 256), int(alpha * 512), stride=1, kernel_size=1),
            *[
                DepthwiseConvEncBlock(int(alpha * 512), int(alpha * 512), stride=1),
                ConvEncBlock(int(alpha * 512), int(alpha * 512), stride=1, kernel_size=1)
                for _ in range(5)
            ],
            DepthwiseConvEncBlock(int(alpha * 512), int(alpha * 512), stride=2),
        ])

        self.block5 = torch.nn.ModuleList([
            ConvEncBlock(int(alpha * 512), int(alpha * 1024), stride=1, kernel_size=1),
            DepthwiseConvEncBlock(int(alpha * 1024), int(alpha * 1024), stride=1),
            ConvEncBlock(int(alpha * 1024), int(alpha * 1024), stride=1, kernel_size=1),
        ])

        self.head = MobilenetEncHead((input_shape[1] // 32, input_shape[2] // 32), int(alpha * 1024), n_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block0(x)
        y = self.block1(y)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)
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
