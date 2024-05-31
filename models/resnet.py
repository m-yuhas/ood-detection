import argparse
import os

import pytorch_lightning
import torch
import torchvision

class ResnetEncBlock(torch.nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, stride):
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

    def forward(self, x):
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

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.act1 = torch.nn.ReLU()
        
        self.conv2 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.act2 = torch.nn.ReLU()

        self.conv_skip = torch.nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0)
        self.bn_skip = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)
        
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act2(y)

        return y + self.bn_skip(self.conv_skip(x))


class ResnetEncMandatory(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn = torch.nn.BatchNorm2d(64)
        self.act = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        return self.pool(y)

class ResnetEncHead(torch.nn.Module):
    def __init__(self, in_dim, in_channels, n_latent):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(in_dim)
        self.fc = torch.nn.Linear(in_channels, n_latent)
        self.in_channels = in_channels

    def forward(self, x):
        y = self.pool(x)
        y = torch.reshape(y, (-1, self.in_channels))
        y = self.fc(y)
        return y
        #return y[:, :y.shape[-1] // 2], y[:, y.shape[-1] // 2:]


class ResnetDecBlock(torch.nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, stride):
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

    def forward(self, x):
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

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(in_channels, out_channels, 3, stride=stride, padding=1, output_padding=stride-1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.act1 = torch.nn.ReLU()

        self.conv2 = torch.nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.act2 = torch.nn.ReLU()

        self.conv_skip = torch.nn.ConvTranspose2d(in_channels, out_channels, 1, stride=stride, padding=0, output_padding=stride-1)
        self.bn_skip = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act2(y)

        return y + self.bn_skip(self.conv_skip(x))

class ResnetDecMandatory(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(64, 3, 7, stride=2, padding=3, output_padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.act = torch.nn.ReLU()
        #self.pool = torch.nn.MaxUnpool2d(3, stride=2, padding=1)
        self.pool = torch.nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1, groups=64)
        self.pool_bn = torch.nn.BatchNorm2d(64)

    def forward(self, x):
        y = self.pool_bn(self.pool(x))
        y = self.conv(y)
        y = self.bn(y)
        return self.act(y)

class ResnetDecHead(torch.nn.Module):
    def __init__(self, n_latent, out_dim, out_channels):
        super().__init__()
        #self.pool = torch.nn.MaxUnpool2d(out_dim)
        self.pool = torch.nn.ConvTranspose2d(out_channels, out_channels, out_dim, stride=1, groups=out_channels)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.fc = torch.nn.Linear(n_latent, out_channels)
        self.out_dim = out_dim,
        self.out_channels = out_channels

    def forward(self, x):
        y = self.fc(x)
        #print((-1, self.out_channels, self.out_dim, self.out_dim))
        y = torch.reshape(y, (-1, self.out_channels, 1, 1))
        return self.bn(self.pool(y))

class Resnet18Enc(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.block0_quant = torch.ao.quantization.QuantStub()
        self.block0_0 = ResnetEncMandatory()
        self.block0_dequant = torch.ao.quantization.DeQuantStub()

        self.block1_quant = torch.ao.quantization.QuantStub()
        self.block1_0 = ResnetEncSmallBlock(64, 64, 1)
        self.block1_1 = ResnetEncSmallBlock(64, 64, 1)
        self.block1_dequant = torch.ao.quantization.DeQuantStub()

        self.block2_quant = torch.ao.quantization.QuantStub()
        self.block2_0 = ResnetEncSmallBlock(64, 128, 2)
        self.block2_1 = ResnetEncSmallBlock(128, 128, 1)
        self.block2_dequant = torch.ao.quantization.DeQuantStub()

        self.block3_quant = torch.ao.quantization.QuantStub()
        self.block3_0 = ResnetEncSmallBlock(128, 256, 2)
        self.block3_1 = ResnetEncSmallBlock(256, 256, 1)
        self.block3_dequant = torch.ao.quantization.DeQuantStub()

        self.block4_quant = torch.ao.quantization.QuantStub()
        self.block4_0 = ResnetEncSmallBlock(256, 512, 2)
        self.block4_1 = ResnetEncSmallBlock(512, 512, 1)

        self.head = ResnetEncHead(7, 512, 500)
        self.mu_dequant = torch.ao.quantization.DeQuantStub()
        self.logvar_dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        y = self.block0_quant(x)
        y = self.block0_0(y)
        y = self.block0_dequant(y)

        y = self.block1_quant(y)
        y = self.block1_0(y)
        y = self.block1_1(y)
        y = self.block1_dequant(y)

        y = self.block2_quant(y)
        y = self.block2_0(y)
        y = self.block2_1(y)
        y = self.block2_dequant(y)

        y = self.block3_quant(y)
        y = self.block3_0(y)
        y = self.block3_1(y)
        y = self.block3_dequant(y)

        y = self.block4_quant(y)
        y = self.block4_0(y)
        y = self.block4_1(y)

        y = self.head(y)
        mu, logvar = y[:, :y.shape[-1] // 2], y[:, y.shape[-1] // 2:]
        return self.mu_dequant(mu), self.logvar_dequant(logvar)


class Resnet18Dec(torch.nn.Module):

    def __init__(self):
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
        #print(y.shape)
        y = self.block3_1(y)
        y = self.block3_0(y)
        #print(y.shape)
        y = self.block2_1(y)
        y = self.block2_0(y)
        #print(y.shape)
        y = self.block1_1(y)
        y = self.block1_0(y)
        #print(y.shape)
        return self.block0_0(y)

class Resnet18Vae(pytorch_lightning.LightningModule):

    def __init__(self, beta=1, learning_rate=1e-5):
        super().__init__()
        self.beta = beta
        self.n_latent = 500
        self.learning_rate = learning_rate
        self.encoder = Resnet18Enc()
        self.decoder = Resnet18Dec()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        stdev = torch.exp(logvar / 2)
        eps = torch.randn_like(stdev)
        z = mu + stdev * eps
        return self.decoder(z), mu, logvar

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, train_batch, batch_idx):
        x, _ = train_batch
        x_hat, mu, logvar = self.forward(x)
        kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1) / self.n_latent
        mse_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='mean')
        loss = mse_loss + self.beta * kl_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, _ = val_batch
        x_hat, mu, logvar = self.forward(x)
        kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1) / self.n_latent
        mse_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='mean')
        loss = mse_loss + self.beta * kl_loss
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        mu, logvar = self.encoder(x)
        kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1) / self.n_latent
        self.log('dkl', kl_loss)
        return kl_loss, y


    def predict_step(self, predict_batch, batch_idx):
        x, y = predict_batch
        mu, logvar = self.encoder(x)
        kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1) / self.n_latent
        return kl_loss, y

class OodDataModule(pytorch_lightning.LightningDataModule):

    def __init__(self, train_path, val_path, test_path, predict_path, batch_size=32):
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self):
        train_set = torchvision.datasets.ImageFolder(
            root=self.hparams.train_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((224, 224), antialias=False)
            ])
        )
        return torch.utils.data.DataLoader(
            train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=len(os.sched_getaffinity(0)),
            drop_last=True)

    def val_dataloader(self):
        val_set = torchvision.datasets.ImageFolder(
            root=self.hparams.val_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((224, 224), antialias=False)
            ])
        )
        return torch.utils.data.DataLoader(
            val_set,
            batch_size=min(len(val_set), self.hparams.batch_size),
            num_workers=len(os.sched_getaffinity(0)),
            drop_last=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(
                root=self.hparams.test_path,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Resize((224, 224), antialias=False)
                ])
            ),
            batch_size=self.hparams.batch_size,
            num_workers=len(os.sched_getaffinity(0)),
            drop_last=True)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(
                root=self.hparams.predict_path,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Resize((224, 224), antialias=False)
                ])
            ),
            batch_size=self.hparams.batch_size,
            num_workers=len(os.sched_getaffinity(0)),
            drop_last=True)

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
        logger=pytorch_lightning.loggers.TensorBoardLogger(save_dir='logs/'),
    )
    trainer.fit(model, datamodule=data)
    torch.save(model, args.name + '.pt')
