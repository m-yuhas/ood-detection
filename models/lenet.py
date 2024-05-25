import torch

class LeNetEncBlock(torch.nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int=5,
                 padding: int=0,
                 dilation: int=1,
                 pooling: int=2,
                 activation=torch.nn.ReLU) -> None:
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.act = activation
        self.pool = torch.nn.MaxPool2d(pooling)

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))

class LeNetDecBlock(torch.nn.Module):
    pass