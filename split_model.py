import argparse
import os
import torch
import torchvision
import pytorch_lightning

from models.vggnet_multihead import VggnetWsvaeMultiHead

class VggBlock0(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.block0_quant = base.encoder.block0_quant
        self.block0 = base.encoder.block0
        self.block0_dequant = base.encoder.block0_dequant
        self.pool0 = base.pool0
        self.enc_head0 = base.enc_head0

    def forward(self, x):
        y = self.block0_quant(x)
        y = self.block0(y)
        y = self.block0_dequant(y)
        l = self.enc_head0(self.pool0(y))
        mu = l[:, :l.shape[-1] // 2]
        return mu[:, 0], y

class VggBlock1(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.block1_quant = base.encoder.block1_quant
        self.block1 = base.encoder.block1
        self.block1_dequant = base.encoder.block1_dequant
        self.pool1 = base.pool1
        self.enc_head1 = base.enc_head1

    def forward(self, x):
        y = self.block1_quant(x)
        y = self.block1(y)
        y = self.block1_dequant(y)
        l = self.enc_head1(self.pool1(y))
        mu = l[:, :l.shape[-1] // 2]
        return mu[:, 0], y

