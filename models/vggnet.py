import argparse
import os

import pytorch_lightning
import torch
import torchvision


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
        self.fc1 = torch.nn.Linear(in_channels * in_dim * in_dim, 4096)
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
        self.fc2 = torch.nn.Linear(4096, out_dim * out_dim * out_channels)
        self.act2 = torch.nn.ReLU()
        self.out_dim = out_dim
        self.out_channels = out_channels

    def forward(self, x):
        y = self.act0(self.fc0(x))
        y = self.act1(self.fc1(y))
        y = self.act2(self.fc2(y))
        y = torch.reshape(y, (-1, self.out_channels, self.out_dim, self.out_dim))
        return y

class Vggnet11Enc(pytorch_lightning.LightningModule):

    def __init__(self):
        super().__init__()
        self.block0_quant = torch.ao.quantization.QuantStub()
        self.block0 = VggnetEncBlock(3, 64, 1)
        self.block0_dequant = torch.ao.quantization.DeQuantStub()

        self.block1_quant = torch.ao.quantization.QuantStub()
        self.block1 = VggnetEncBlock(64, 128, 1)
        self.block1_dequant = torch.ao.quantization.DeQuantStub()

        self.block2_quant = torch.ao.quantization.QuantStub()
        self.block2 = VggnetEncBlock(128, 256, 2)
        self.block2_dequant = torch.ao.quantization.DeQuantStub()

        self.block3_quant = torch.ao.quantization.QuantStub()
        self.block3 = VggnetEncBlock(256, 512, 2)
        self.block3_dequant = torch.ao.quantization.DeQuantStub()

        self.block4_quant = torch.ao.quantization.QuantStub()
        self.block4 = VggnetEncBlock(512, 512, 2)
        self.block4_dequant = torch.ao.quantization.DeQuantStub()


    def forward(self, x):
        y = self.block0_quant(x)
        y = self.block0(y)
        y = self.block0_dequant(y)

        y = self.block1_quant(y)
        y = self.block1(y)
        y = self.block1_dequant(y)

        y = self.block2_quant(y)
        y = self.block2(y)
        y = self.block2_dequant(y)

        y = self.block3_quant(y)
        y = self.block3(y)
        y = self.block3_dequant(y)

        y = self.block4_quant(y)
        y = self.block4(y)
        y = self.block4_dequant(y)

        return y 

class Vggnet13Enc(pytorch_lightning.LightningModule):

    def __init__(self):
        super().__init__()
        self.block0_quant = torch.ao.quantization.QuantStub()
        self.block0 = VggnetEncBlock(3, 64, 2)
        self.block0_dequant = torch.ao.quantization.DeQuantStub()

        self.block1_quant = torch.ao.quantization.QuantStub()
        self.block1 = VggnetEncBlock(64, 128, 2)
        self.block1_dequant = torch.ao.quantization.DeQuantStub()

        self.block2_quant = torch.ao.quantization.QuantStub()
        self.block2 = VggnetEncBlock(128, 256, 2)
        self.block2_dequant = torch.ao.quantization.DeQuantStub()

        self.block3_quant = torch.ao.quantization.QuantStub()
        self.block3 = VggnetEncBlock(256, 512, 2)
        self.block3_dequant = torch.ao.quantization.DeQuantStub()

        self.block4_quant = torch.ao.quantization.QuantStub()
        self.block4 = VggnetEncBlock(512, 512, 2)
        self.block4_dequant = torch.ao.quantization.DeQuantStub()


    def forward(self, x):
        y = self.block0_quant(x)
        y = self.block0(y)
        y = self.block0_dequant(y)

        y = self.block1_quant(y)
        y = self.block1(y)
        y = self.block1_dequant(y)

        y = self.block2_quant(y)
        y = self.block2(y)
        y = self.block2_dequant(y)

        y = self.block3_quant(y)
        y = self.block3(y)
        y = self.block3_dequant(y)

        y = self.block4_quant(y)
        y = self.block4(y)
        y = self.block4_dequant(y)

        return y 


class Vggnet16Enc(pytorch_lightning.LightningModule):

    def __init__(self):
        super().__init__()
        self.block0_quant = torch.ao.quantization.QuantStub()
        self.block0 = VggnetEncBlock(3, 64, 2)
        self.block0_dequant = torch.ao.quantization.DeQuantStub()

        self.block1_quant = torch.ao.quantization.QuantStub()
        self.block1 = VggnetEncBlock(64, 128, 2)
        self.block1_dequant = torch.ao.quantization.DeQuantStub()

        self.block2_quant = torch.ao.quantization.QuantStub()
        self.block2 = VggnetEncBlock(128, 256, 3)
        self.block2_dequant = torch.ao.quantization.DeQuantStub()

        self.block3_quant = torch.ao.quantization.QuantStub()
        self.block3 = VggnetEncBlock(256, 512, 3)
        self.block3_dequant = torch.ao.quantization.DeQuantStub()

        self.block4_quant = torch.ao.quantization.QuantStub()
        self.block4 = VggnetEncBlock(512, 512, 3)
        self.block4_dequant = torch.ao.quantization.DeQuantStub()


    def forward(self, x):
        y = self.block0_quant(x)
        y = self.block0(y)
        y = self.block0_dequant(y)

        y = self.block1_quant(y)
        y = self.block1(y)
        y = self.block1_dequant(y)

        y = self.block2_quant(y)
        y = self.block2(y)
        y = self.block2_dequant(y)

        y = self.block3_quant(y)
        y = self.block3(y)
        y = self.block3_dequant(y)

        y = self.block4_quant(y)
        y = self.block4(y)
        y = self.block4_dequant(y)

        return y 


class Vggnet19Enc(pytorch_lightning.LightningModule):

    def __init__(self):
        super().__init__()
        self.block0_quant = torch.ao.quantization.QuantStub()
        self.block0 = VggnetEncBlock(3, 64, 2)
        self.block0_dequant = torch.ao.quantization.DeQuantStub()

        self.block1_quant = torch.ao.quantization.QuantStub()
        self.block1 = VggnetEncBlock(64, 128, 2)
        self.block1_dequant = torch.ao.quantization.DeQuantStub()

        self.block2_quant = torch.ao.quantization.QuantStub()
        self.block2 = VggnetEncBlock(128, 256, 4)
        self.block2_dequant = torch.ao.quantization.DeQuantStub()

        self.block3_quant = torch.ao.quantization.QuantStub()
        self.block3 = VggnetEncBlock(256, 512, 4)
        self.block3_dequant = torch.ao.quantization.DeQuantStub()

        self.block4_quant = torch.ao.quantization.QuantStub()
        self.block4 = VggnetEncBlock(512, 512, 4)
        self.block4_dequant = torch.ao.quantization.DeQuantStub()


    def forward(self, x):
        y = self.block0_quant(x)
        y = self.block0(y)
        y = self.block0_dequant(y)

        y = self.block1_quant(y)
        y = self.block1(y)
        y = self.block1_dequant(y)

        y = self.block2_quant(y)
        y = self.block2(y)
        y = self.block2_dequant(y)

        y = self.block3_quant(y)
        y = self.block3(y)
        y = self.block3_dequant(y)

        y = self.block4_quant(y)
        y = self.block4(y)
        y = self.block4_dequant(y)

        return y 


class Vggnet11Dec(pytorch_lightning.LightningModule):

    def __init__(self, n_latent):
        super().__init__()
        self.head = VggnetDecHead(n_latent, 7, 512)

        self.block4 = VggnetDecBlock(512, 512, 2)
        self.block3 = VggnetDecBlock(512, 256, 2)
        self.block2 = VggnetDecBlock(256, 128, 2)
        self.block1 = VggnetDecBlock(128, 64, 1)
        self.block0 = VggnetDecBlock(64, 3, 1)

    def forward(self, x):
        y = self.head(x)
        y = self.block4(y)
        y = self.block3(y)
        y = self.block2(y)
        y = self.block1(y)
        y = self.block0(y)
        return y


class Vggnet13Dec(pytorch_lightning.LightningModule):

    def __init__(self, n_latent):
        super().__init__()
        self.head = VggnetDecHead(n_latent, 7, 512)

        self.block4 = VggnetDecBlock(512, 512, 2)
        self.block3 = VggnetDecBlock(512, 256, 2)
        self.block2 = VggnetDecBlock(256, 128, 2)
        self.block1 = VggnetDecBlock(128, 64, 2)
        self.block0 = VggnetDecBlock(64, 3, 2)

    def forward(self, x):
        y = self.head(x)
        y = self.block4(y)
        y = self.block3(y)
        y = self.block2(y)
        y = self.block1(y)
        y = self.block0(y)
        return y

class Vggnet16Dec(pytorch_lightning.LightningModule):

    def __init__(self, n_latent):
        super().__init__()
        self.head = VggnetDecHead(n_latent, 7, 512)

        self.block4 = VggnetDecBlock(512, 512, 3)
        self.block3 = VggnetDecBlock(512, 256, 3)
        self.block2 = VggnetDecBlock(256, 128, 3)
        self.block1 = VggnetDecBlock(128, 64, 2)
        self.block0 = VggnetDecBlock(64, 3, 2)

    def forward(self, x):
        y = self.head(x)
        y = self.block4(y)
        y = self.block3(y)
        y = self.block2(y)
        y = self.block1(y)
        y = self.block0(y)
        return y

class Vggnet19Dec(pytorch_lightning.LightningModule):

    def __init__(self, n_latent):
        super().__init__()
        self.head = VggnetDecHead(n_latent, 7, 512)

        self.block4 = VggnetDecBlock(512, 512, 4)
        self.block3 = VggnetDecBlock(512, 256, 4)
        self.block2 = VggnetDecBlock(256, 128, 4)
        self.block1 = VggnetDecBlock(128, 64, 2)
        self.block0 = VggnetDecBlock(64, 3, 2)

    def forward(self, x):
        y = self.head(x)
        y = self.block4(y)
        y = self.block3(y)
        y = self.block2(y)
        y = self.block1(y)
        y = self.block0(y)
        return y


class VggnetVae(pytorch_lightning.LightningModule):

    def __init__(self, depth=11, n_latent=250, beta=1, input_dim=(224, 224), learning_rate=1e-5):
        super().__init__()
        self.beta = beta
        self.n_latent = n_latent
        self.learning_rate = learning_rate
        if depth == 11:
            self.encoder = Vggnet11Enc()
            self.decoder = Vggnet11Dec(n_latent)
        elif depth == 13:
            self.encoder = Vggnet13Enc()
            self.decoder = Vggnet13Dec(n_latent)
        elif depth == 16:
            self.encoder = Vggnet16Enc()
            self.decoder = Vggnet16Dec(n_latent)
        elif depth == 19:
            self.encoder = Vggnet19Enc()
            self.decoder = Vggnet19Dec(n_latent)
        self.enc_head = VggnetEncHead(input_dim[0] // 32, 512, 2 * n_latent)

    def forward(self, x):
        y = self.enc_head(self.encoder(x))
        mu, logvar = y[:, :y.shape[-1] // 2], y[:, y.shape[-1] // 2:]
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

class VggnetWsvae(pytorch_lightning.LightningModule):

    def __init__(self, depth=11, n_latent=250, levels=5, alpha=1, beta=1, gamma=1, input_dim=(224, 224), learning_rate=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_latent = n_latent
        self.learning_rate = learning_rate
        self.levels = levels
        if depth == 11:
            self.encoder = Vggnet11Enc()
            self.decoder = Vggnet11Dec(n_latent)
        elif depth == 13:
            self.encoder = Vggnet13Enc()
            self.decoder = Vggnet13Dec(n_latent)
        elif depth == 16:
            self.encoder = Vggnet16Enc()
            self.decoder = Vggnet16Dec(n_latent)
        elif depth == 19:
            self.encoder = Vggnet19Enc()
            self.decoder = Vggnet19Dec(n_latent)
        self.enc_head = VggnetEncHead(input_dim[0] // 32, 512, 2 * n_latent)
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(1, 4096),
            torch.nn.Sigmoid(),
            torch.nn.Linear(4096, 4096),
            torch.nn.Sigmoid(),
            torch.nn.Linear(4096, levels),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.enc_head(self.encoder(x))
        mu, logvar = y[:, :y.shape[-1] // 2], y[:, y.shape[-1] // 2:]
        y = self.cls_head(torch.unsqueeze(mu[:, 0],1))
        stdev = torch.exp(logvar / 2)
        eps = torch.randn_like(stdev)
        z = mu + stdev * eps
        return self.decoder(z), mu, logvar, y

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x_hat, mu, logvar, y_hat = self.forward(x)
        kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1) / self.n_latent
        mse_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='mean')
        ce_loss = torch.nn.functional.binary_cross_entropy(y_hat, torch.nn.functional.one_hot(y, self.levels).float(), reduction='mean')
        pstn_loss = torch.nn.functional.mse_loss(mu[:, 0], (y - self.levels / 2) / self.levels, reduction='mean')
        loss = mse_loss + self.beta * kl_loss + self.alpha * ce_loss + self.gamma * pstn_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x_hat, mu, logvar, y_hat = self.forward(x)
        kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1) / self.n_latent
        mse_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='mean')
        ce_loss = torch.nn.functional.binary_cross_entropy(y_hat, torch.nn.functional.one_hot(y, self.levels).float(), reduction='mean')
        pstn_loss = torch.nn.functional.mse_loss(mu[:, 0], (y - self.levels / 2) / self.levels, reduction='mean')
        loss = mse_loss + self.beta * kl_loss + self.alpha * ce_loss + self.gamma * pstn_loss
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        mu, logvar = self.encoder(x)
        #kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1) / self.n_latent
        #self.log('dkl', kl_loss)
        self.log('oodscore', mu[:,0])
        return mu[:, 0], y


    def predict_step(self, predict_batch, batch_idx):
        x, y = predict_batch
        mu, logvar = self.encoder(x)
        #kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1) / self.n_latent
        return mu[:, 0], y
