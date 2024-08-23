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
        self.enc_quant0 = torch.quantization.QuantStub()
        self.enc_head0 = base.enc_head0
        self.enc_dequant0 = torch.quantization.DeQuantStub()

    def forward(self, x):
        y = self.block0_quant(x)
        y = self.block0(y)
        l = self.enc_dequant0(self.enc_head0(self.pool0(y)))
        y = self.block0_dequant(y)
        mu = l[:, :l.shape[-1] // 2]
        return mu[:, 0], y

class VggBlock1(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.block1_quant = base.encoder.block1_quant
        self.block1 = base.encoder.block1
        self.block1_dequant = base.encoder.block1_dequant
        self.pool1 = base.pool1
        self.enc_quant1 = torch.quantization.QuantStub()
        self.enc_head1 = base.enc_head1
        self.enc_dequant1 = torch.quantization.DeQuantStub()

    def forward(self, x):
        y = self.block1_quant(x)
        y = self.block1(y)
        l = self.enc_dequant1(self.enc_head1(self.pool1(y)))
        y = self.block1_dequant(y)
        mu = l[:, :l.shape[-1] // 2]
        return mu[:, 0], y

class VggBlock2(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.block2_quant = base.encoder.block2_quant
        self.block2 = base.encoder.block2
        self.block2_dequant = base.encoder.block2_dequant
        self.pool2 = base.pool2
        self.enc_quant2 = torch.quantization.QuantStub()
        self.enc_head2 = base.enc_head2
        self.enc_dequant2 = torch.quantization.DeQuantStub()

    def forward(self, x):
        y = self.block2_quant(x)
        y = self.block2(y)
        l = self.enc_dequant2(self.enc_head2(self.pool2(y)))
        y = self.block2_dequant(y)
        mu = l[:, :l.shape[-1] // 2]
        return mu[:, 0], y

class VggBlock3(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.block3_quant = base.encoder.block3_quant
        self.block3 = base.encoder.block3
        self.block3_dequant = base.encoder.block3_dequant
        self.pool3 = base.pool3
        self.enc_quant3 = torch.quantization.QuantStub()
        self.enc_head3 = base.enc_head3
        self.enc_dequant3 = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        y = self.block3_quant(x)
        y = self.block3(y)
        l = self.enc_dequant3(self.enc_head3(self.pool3(y)))
        mu = l[:, :l.shape[-1] // 2]
        y = self.block3_dequant(y)
        return mu[:, 0], y

class VggBlock4(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.block4_quant = base.encoder.block4_quant
        self.block4 = base.encoder.block4
        self.block4_dequant = base.encoder.block4_dequant
        self.enc_quant4 = torch.quantization.QuantStub()
        self.enc_head4 = base.enc_head4
        self.enc_dequant4 = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        y = self.block4_quant(x)
        y = self.block4(y)
        l = self.enc_dequant4(self.enc_head4(y))
        mu = l[:, :l.shape[-1] // 2]
        y = self.block4_dequant(x)
        return mu[:, 0], y

def split_model(base: str):
    base = torch.load(base)
    return [
        VggBlock0(base),
        VggBlock1(base),
        VggBlock2(base),
        VggBlock3(base),
        VggBlock4(base),
    ]

def static_quantize(calset, blocks):
    # Eval Mode
    blocks = [b.to('cpu').eval() for b in blocks]
 
    # Fuse layers
    block0_fuse_list = [
        ['block0.conv.conv0', 'block0.conv.bn0', 'block0.conv.act0'],
        ['block0.conv.conv1', 'block0.conv.bn1', 'block0.conv.act1'],
        ['enc_head0.fc1', 'enc_head0.act1'],
        ['enc_head0.fc2', 'enc_head0.act2'],
    ]
    blocks[0] = torch.quantization.fuse_modules(blocks[0], block0_fuse_list)
    block1_fuse_list = [
        ['block1.conv.conv0', 'block1.conv.bn0', 'block1.conv.act0'],
        ['block1.conv.conv1', 'block1.conv.bn1', 'block1.conv.act1'],
        ['enc_head1.fc1', 'enc_head1.act1'],
        ['enc_head1.fc2', 'enc_head1.act2'],
    ]
    blocks[1] = torch.quantization.fuse_modules(blocks[1], block1_fuse_list)
    block2_fuse_list = [
        ['block2.conv.conv0', 'block2.conv.bn0', 'block2.conv.act0'],
        ['block2.conv.conv1', 'block2.conv.bn1', 'block2.conv.act1'],
        ['enc_head2.fc1', 'enc_head2.act1'],
        ['enc_head2.fc2', 'enc_head2.act2'],
    ]
    blocks[2] = torch.quantization.fuse_modules(blocks[2], block2_fuse_list)
    block3_fuse_list = [
        ['block3.conv.conv0', 'block3.conv.bn0', 'block3.conv.act0'],
        ['block3.conv.conv1', 'block3.conv.bn1', 'block3.conv.act1'],
        ['enc_head3.fc1', 'enc_head3.act1'],
        ['enc_head3.fc2', 'enc_head3.act2'],
    ]
    blocks[3] = torch.quantization.fuse_modules(blocks[3], block3_fuse_list)   
    block4_fuse_list = [
        ['block4.conv.conv0', 'block4.conv.bn0', 'block4.conv.act0'],
        ['block4.conv.conv1', 'block4.conv.bn1', 'block4.conv.act1'],
        ['enc_head4.fc1', 'enc_head4.act1'],
        ['enc_head4.fc2', 'enc_head4.act2'],
    ]
    blocks[4] = torch.quantization.fuse_modules(blocks[4], block4_fuse_list)

    # Set QConfig
    for block in blocks:
        block.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    blocks = [torch.quantization.prepare(b) for b in blocks]

    # Calibrate with train set
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ])
    cal_data = torchvision.datasets.ImageFolder(
        calset,
        transform=transforms,
        is_valid_file =lambda x: True if x.endswith('.png') else False
    )
    data_loader = torch.utils.data.DataLoader(cal_data, batch_size=16, shuffle=True)
    with torch.no_grad():
        count = 0
        for x, _ in data_loader:
            print(f'Calibration Batch: {count}')
            count += 1
            _, x = blocks[0](x)
            _, x = blocks[1](x)
            _, x = blocks[2](x)
            _, x = blocks[3](x)
            _ = blocks[4](x)

    # Convert to quantized model
    blocks = [torch.quantization.convert(b) for b in blocks]
    return blocks

def save_gpu_blocks(blocks):
    x = torch.randn(1,3,224,224).to(torch.device('cuda'))
    for idx, block in enumerate(blocks):
        block = block.to(torch.device('cuda'))
        block = block.eval()
        b = torch.jit.trace(block, (x))
        torch.jit.save(b, f'vggnetblock{idx}.pt')
        if idx < 4:
            _, x = block(x)

def save_cpu_blocks(blocks):
    x = torch.randn(1,3,224,224).to(torch.device('cpu'))
    for idx, block in enumerate(blocks):
        print(f'Proc. Block. {idx}')
        block = block.to(torch.device('cpu'))
        block = block.eval()
        b = torch.jit.trace(block, (x))
        torch.jit.save(b, f'vggnetblock{idx}q.pt')
        if idx < 4:
            _, x = block(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Quantized OOD Detector')
    parser.add_argument(
        '--base',
        help='Path the base model',
    )
    parser.add_argument(
        '--calset',
        help='Path to calibration dataset',
    )
    args = parser.parse_args()
    print('Splitting model...')
    blocks = split_model(args.base)
    print('Saving blocks...')
    save_gpu_blocks(blocks)
    print('Quantizing...')
    q_blocks = static_quantize(args.calset, blocks)
    print('Saving quantized blocks...')
    save_cpu_blocks(q_blocks)