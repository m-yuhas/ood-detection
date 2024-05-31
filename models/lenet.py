from typing import Callable, Dict, Tuple
import math
import torch

LENET1 = [
    {'out_channels': 4},
    {'out_channels': 12},
]

LENET4 = [
    {'out_channels': 4},
    {'out_channels': 16},
]

LENET5 = [
    {'out_channels': 6},
    {'out_channels': 16},
]

DAVE2 = [
    {'out_channels': 24},
    {'out_channels': 36},
    {'out_channels': 48},
    {'out_channels': 64, 'conv_kernel': 3, 'pool_kernel': 1},
    {'out_channels': 65, 'conv_kernel': 3, 'pool_kernel': 1},
]

ALEXNET = [
    {'out_channels': 96, 'conv_kernel': 11, 'conv_stride': 4, 'pool_kernel': 3, 'pool_stride': 2},
    {'out_channels': 256, 'conv_pad': 2, 'pool_kernel': 3, 'pool_stride': 2},
    {'out_channels': 384, 'conv_kernel': 3, 'conv_pad': 1},
    {'out_channels': 384, 'conv_kernel': 3, 'conv_pad': 1},
    {'out_channels': 256, 'conv_kernel': 3, 'conv_pad': 1, 'pool_kernel': 3, 'pool_stride': 2},
]

class LeNetEncBlock(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, **kwargs: Dict) -> None:
        defaults = {
            'conv_kernel': 5,
            'conv_pad': 0,
            'conv_dilation': 1,
            'conv_stride': 2,
            'pool_kernel': 2,
            'pool_pad': 0,
            'pool_dilation': 1,
            'pool_stride': 2,
            'activation': torch.nn.ReLU()
        }
        for arg in defaults:
            if arg not in kwargs:
                kwargs[arg] = defaults[arg]
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kwargs['conv_kernel'],
            padding=kwargs['conv_pad'],
            dilation=kwargs['conv_dilation'],
            stride=kwargs['conv_stride']
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.act = kwargs['activation']
        if kwargs['pool_kernel'] == 1:
            self.pool = torch.nn.Identity()
        else:    
            self.pool = torch.nn.MaxPool2d(
                kwargs['pool_kernel'],
                stride=['pool_stride'],
                padding=['pool_pad'],
                dilation=['pool_dilation']
            )
        conv_out_dim = lambda x : math.floor(
            x + \
            2 * kwargs['padding'] - \
            kwargs['dilation'] * (kwargs['kernel_size'] - 1) \
        )
        pool_out_dim = lambda x : math.floor(
            (x - (kwargs['pooling'] - 1) - 1) / kwargs['pooling'] + 1
        )
        self.get_output_size = lambda c, h, w : (
            out_channels,
            pool_out_dim(conv_out_dim(h)),
            pool_out_dim(conv_out_dim(w)),
        )

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class LeNetDecBlock(torch.nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Dict) -> None:
        defaults = {
            'kernel_size': 5,
            'padding': 0,
            'dilation': 1,
            'unpooling': 2,
            'activation': torch.nn.ReLU()
        }
        for arg in defaults:
            if arg not in kwargs:
                kwargs[arg] = defaults[arg]
        if kwargs['unpooling'] == 1:
            self.unpooling = torch.nn.Identity()
        else:
            self.unpool = torch.nn.Upsample(kwargs['unpooling'])
        adjusted_padding = kwargs['dilation'] * \
            (kwargs['kernel_size'] - 1) - kwargs['padding']
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kwargs['kernel_size'],
            padding=adjusted_padding,
            dilation=kwargs['dilation'],
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.act = kwargs['activation']

    def forward(self, x):
        return self.act(self.bn(self.conv(self.unpool)))


class LeNetEnc(torch.nn.Module):

    def __init__(self,
                 variant: int=1,
                 input_shape: Tuple[int] = (1, 28, 28),
                 config: Dict=None):
        if variant == 1:
            config = LENET1
        elif variant == 4:
            config = LENET4
        elif variant == 5:
            config = LENET5
        elif config is None:
            raise ValueError("Valid LeNet variants: 1, 4, 5, else config required.")

        self.blocks = torch.nn.ModuleList()
        self.block_outputs = []
        for block in config:
            self.blocks.append(LeNetEncBlock(
                input_shape[0],
                block['out_channels'],
                block,
            ))
            input_shape = self.blocks[-1].get_output_size(input_shape)
            self.block_outputs.append(input_shape)


    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class LeNetDec(torch.nn.Module):

    def __init__(self,
                 block_inputs: Tuple[Tuple[int]],
                 variant: int=1,
                 config: Dict=None):
        if variant == 1:
            config = LENET1
        elif variant == 4:
            config = LENET4
        elif variant == 5:
            config = LENET5
        elif config is None:
            raise ValueError("Valid LeNet variants: 1, 4, 5, else config required.")

        self.blocks = torch.nn.ModuleList()
        for block, input_shape in zip(config[::-1], block_inputs):
            block['unpooling'] = (input_shape[1], input_shape[2])
            self.blocks.append(LeNetDecBlock(
                input_shape[0],
                block
            ))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class Dave2Enc(torch.nn.Module):

    def __init__(self, input_shape: Tuple[int] = (1, 28, 28)):
        self.blocks = torch.nn.ModuleList()
        self.block_outputs = []
        for block in DAVE2:
            self.blocks.append(LeNetEncBlock(
                input_shape[0],
                block['out_channels'],
                block,
            ))
            input_shape = self.blocks[-1].get_output_size(input_shape)
            self.block_outputs.append(input_shape)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class Dave2Dec(torch.nn.Module):

    def __init__(self, block_inputs: Tuple[Tuple[int]]):
        self.blocks = torch.nn.ModuleList()
        for block, input_shape in zip(DAVE2[::-1], block_inputs):
            block['unpooling'] = (input_shape[1], input_shape[2])
            self.blocks.append(LeNetDecBlock(
                input_shape[0],
                block
            ))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
