from typing import Callable, Dict, List, Tuple

import math
import torch

LENET1 = [
    {'out_channels': 4},
    {'out_channels': 12},
]

LENET4 = [
    {'out_channels': 4},
    {'out_channels': 16},
    {'fully_connected': 120}
]

LENET5 = [
    {'out_channels': 6},
    {'out_channels': 16},
    {'fully_connected': 120},
    {'fully_connected': 84},
]

DAVE2 = [
    {'out_channels': 24},
    {'out_channels': 36},
    {'out_channels': 48},
    {'out_channels': 64, 'conv_kernel': 3, 'pool_kernel': 1},
    {'out_channels': 65, 'conv_kernel': 3, 'pool_kernel': 1},
    {'fully_connected': 1164},
    {'fully_connected': 100},
    {'fully_connected': 50},
    {'fully_connected': 10},
]

ALEXNET = [
    {'out_channels': 96, 'conv_kernel': 11, 'conv_stride': 4, 'pool_kernel': 3, 'pool_stride': 2},
    {'out_channels': 256, 'conv_pad': 2, 'pool_kernel': 3, 'pool_stride': 2},
    {'out_channels': 384, 'conv_kernel': 3, 'conv_pad': 1},
    {'out_channels': 384, 'conv_kernel': 3, 'conv_pad': 1},
    {'out_channels': 256, 'conv_kernel': 3, 'conv_pad': 1, 'pool_kernel': 3, 'pool_stride': 2},
    {'fully_connected': 4096},
    {'fully_connected': 4096},
]

class ConvNetEncBlock(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, **kwargs: Dict) -> None:
        super().__init__()
        defaults = {
            'conv_kernel': 5,
            'conv_pad': 0,
            'conv_dilation': 1,
            'conv_stride': 1,
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
                stride=kwargs['pool_stride'],
                padding=kwargs['pool_pad'],
                dilation=kwargs['pool_dilation']
            )
        conv_out_dim = lambda x : math.floor(
            x + \
            2 * kwargs['conv_pad'] - \
            kwargs['conv_dilation'] * (kwargs['conv_kernel'] - 1) \
        )
        pool_out_dim = lambda x : math.floor(
            (x - (kwargs['pool_kernel'] - 1) - 1) / kwargs['pool_stride'] + 1
        )
        self.get_output_size = lambda c, h, w : (
            out_channels,
            pool_out_dim(conv_out_dim(h)),
            pool_out_dim(conv_out_dim(w)),
        )

    def forward(self, x):
        print(x.shape)
        return self.pool(self.act(self.bn(self.conv(x))))


class ConvNetDecBlock(torch.nn.Module):
    
    def __init__(self, in_channels: int, out_shape: Tuple[int], out_channels: int, **kwargs: Dict) -> None:
        super().__init__()
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
        #if kwargs['unpooling'] == 1:
        #    self.unpooling = torch.nn.Identity()
        #else:
        self.unpool = torch.nn.Upsample(out_shape)
        #adjusted_padding = kwargs['dilation'] * \
        #    (kwargs['kernel_size'] - 1) - kwargs['padding']
        adjusted_padding = kwargs['kernel_size'] // 2
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
        return self.act(self.bn(self.conv(self.unpool(x))))


class ConvNetEnc(torch.nn.Module):

    def __init__(self,
                 layers: List[Dict],
                 n_latent: int=10,
                 input_shape: Tuple[int] = (1, 28, 28)):
        super().__init__()
        self.conv_blocks = torch.nn.Sequential()
        self.block_outputs = [input_shape]
        last_idx = 0
        for i, layer in enumerate(layers):
            if 'fully_connected' not in layer:
                self.conv_blocks.append(ConvNetEncBlock(
                    input_shape[0],
                    **layer,
                ))
                input_shape = self.conv_blocks[-1].get_output_size(*input_shape)
                self.block_outputs.append(input_shape)
            else:
                last_idx = i
                break
        self.flatten = torch.nn.Flatten()
        self.block_outputs.append([prod := 1, [prod := prod * dim for dim in input_shape]][-1][-1])
        self.fc_blocks = torch.nn.Sequential()
        while last_idx < len(layers):
                self.fc_blocks.append(torch.nn.Sequential(
                    torch.nn.Linear(self.block_outputs[-1], layers[last_idx]['fully_connected']),
                    torch.nn.ReLU(), # TODO: support multiple activation functions
                ))
                self.block_outputs.append(layers[last_idx]['fully_connected']),
                last_idx += 1
        self.final = torch.nn.Linear(self.block_outputs[-1], n_latent)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.flatten(x)
        x = self.fc_blocks(x)
        return self.final(x)


class ConvNetDec(torch.nn.Module):

    def __init__(self,
                 layers: List[Dict],
                 block_outputs: Tuple[Tuple[int]],
                 n_latent: int=10):
        super().__init__()
        self.n_latent = n_latent
        self.fc_blocks = torch.nn.Sequential()
        block_outputs.append(n_latent)
        block_outputs = block_outputs[::-1]
        last_idx = 1
        while isinstance(block_outputs[last_idx], int):
            self.fc_blocks.append(torch.nn.Sequential(
                torch.nn.Linear(block_outputs[last_idx - 1], block_outputs[last_idx]),
                torch.nn.ReLU(),
            ))
            last_idx += 1
        self.reshape_dim = block_outputs[last_idx]
        self.conv_blocks = torch.nn.Sequential()
        layers = layers[::-1][last_idx - 2:]
        block_outputs = block_outputs[last_idx:]
        print(layers)
        print(block_outputs)
        for idx, (block, input_shape) in enumerate(zip(layers, block_outputs[:-1])):
            block['unpooling'] = (input_shape[1], input_shape[2])
            block.pop('out_channels')
            self.conv_blocks.append(ConvNetDecBlock(
                input_shape[0],
                block_outputs[idx + 1][1:],
                block_outputs[idx + 1][0],
                **block
            ))
        print(self.conv_blocks)

    def forward(self, x):
        x = self.fc_blocks(x)
        x = torch.reshape(x, (-1, *self.reshape_dim))
        return self.conv_blocks(x)


class LeNetEnc(torch.nn.Module):

    def __init__(self,
                 variant: int=1,
                 n_latent: int=10,
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

        self.blocks = torch.nn.Sequential()
        self.block_outputs = []
        for block in config:
            if 'fully_connected' not in block:
                self.blocks.append(ConvNetEncBlock(
                    input_shape[0],
                    block['out_channels'],
                    block,
                ))
                input_shape = self.blocks[-1].get_output_size(input_shape)
                self.block_outputs.append(input_shape)
            else:
                self.blocks.append(torch.nn.Flatten())
                self.blocks.append(torch.nn.Linear(input_shape[0] * input_shape[1] * input_shape[2], block['fully_connected']))
                self.blocks.append(torch.nn.ReLU())
                input_shape = (block['fully_connected'],)
        self.blocks.append(torch.nn.Linear(input_shape, n_latent))

    def forward(self, x):
        return self.blocks(x)


class LeNetDec(torch.nn.Module):

    def __init__(self,
                 block_inputs: Tuple[Tuple[int]],
                 n_latent: int=10,
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

        self.blocks = torch.nn.Sequential()
        for block, input_shape in zip(config[::-1], block_inputs):
            block['unpooling'] = (input_shape[1], input_shape[2])
            self.blocks.append(ConvNetDecBlock(
                input_shape[0],
                block
            ))

    def forward(self, x):
        return self.blocks(x)

class DaveEnc(torch.nn.Module):

    def __init__(self, input_shape: Tuple[int] = (1, 28, 28)):
        self.blocks = torch.nn.Sequential()
        self.block_outputs = []
        for block in DAVE2:
            self.blocks.append(ConvNetEncBlock(
                input_shape[0],
                block['out_channels'],
                block,
            ))
            input_shape = self.blocks[-1].get_output_size(input_shape)
            self.block_outputs.append(input_shape)

    def forward(self, x):
        return self.blocks(x)

class DaveDec(torch.nn.Module):

    def __init__(self, block_inputs: Tuple[Tuple[int]]):
        self.blocks = torch.nn.Sequential()
        for block, input_shape in zip(DAVE2[::-1], block_inputs):
            block['unpooling'] = (input_shape[1], input_shape[2])
            self.blocks.append(ConvNetDecBlock(
                input_shape[0],
                block
            ))

    def forward(self, x):
        return self.blocks(x)
