import argparse
import os

import pytorch_lightning
import torch
import torchvision

from vggnet import * 


class VggnetEncHeadSmall(pytorch_lightning.LightningModule):
    def __init__(self, in_dim, in_channels, n_latent):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels * in_dim * in_dim, 4096)
        self.act1 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(4096, n_latent)
        self.in_size = in_channels * in_dim * in_dim

    def forward(self, x):
        y = torch.reshape(x, (-1, self.in_size))
        y = self.act1(self.fc1(y))
        y = self.fc3(y)
        return y[:, :y.shape[-1] // 2], y[:, y.shape[-1] // 2:]


class VggnetDecHeadSmall(pytorch_lightning.LightningModule):
    def __init__(self, n_latent, out_dim, out_channels):
        super().__init__()
        self.fc0 = torch.nn.Linear(n_latent, 4096)
        self.act0 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(4096, out_dim * out_dim * out_channels)
        self.act2 = torch.nn.ReLU()
        self.out_dim = out_dim
        self.out_channels = out_channels

    def forward(self, x):
        y = self.act0(self.fc0(x))
        y = self.act2(self.fc2(y))
        y = torch.reshape(y, (-1, self.out_channels, self.out_dim, self.out_dim))
        return y

class VggnetVaeMultiHead(pytorch_lightning.LightningModule):

    def __init__(self, base_model, n_latent=500, beta=1, learning_rate=1e-5):
        super().__init__()
        self.beta = beta
        self.n_latent = n_latent
        self.learning_rate = learning_rate
        m = torch.load(base_model)
        self.encoder = m.encoder
        self.decoder = m.decoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        #self.encoder.freeze()
        #self.decoder.freeze()
        self.pool0 = torch.nn.MaxPool2d(16)
        self.pool1 = torch.nn.MaxPool2d(8)
        self.pool2 = torch.nn.MaxPool2d(4)
        self.pool3 = torch.nn.MaxPool2d(2)
        self.enc_head0 = VggnetEncHead(7, 64, n_latent)
        self.enc_head1 = VggnetEncHead(7, 128, n_latent)
        self.enc_head2 = VggnetEncHead(7, 256, n_latent)
        self.enc_head3 = VggnetEncHead(7, 512, n_latent)
        self.enc_head4 = m.enc_head
        for param in self.enc_head4.parameters():
            param.requires_grad = False
        #self.enc_head4.freeze()
        self.dec_head0 = VggnetDecHead(n_latent // 2, 7, 64)
        self.dec_head1 = VggnetDecHead(n_latent // 2, 7, 128)
        self.dec_head2 = VggnetDecHead(n_latent // 2, 7, 256)
        self.dec_head3 = VggnetDecHead(n_latent // 2, 7, 512)
        self.dec_head4 = m.decoder.head
        for param in self.dec_head4.parameters():
            param.requires_grad = False
        #self.dec_head4.freeze()
        self.unpool0 = torch.nn.Upsample(scale_factor=16)
        self.unpool1 = torch.nn.Upsample(scale_factor=8)
        self.unpool2 = torch.nn.Upsample(scale_factor=4)
        self.unpool3 = torch.nn.Upsample(scale_factor=2)

    def forward_enc(self, x):
        y0 = self.encoder.block0_quant(x)
        y0 = self.encoder.block0(y0)
        y0 = self.encoder.block0_dequant(y0)
        mu0, logvar0 = self.enc_head0(self.pool0(y0))

        y1 = self.encoder.block1_quant(y0)
        y1 = self.encoder.block1(y1)
        y1 = self.encoder.block1_dequant(y1)
        mu1, logvar1 = self.enc_head1(self.pool1(y1))

        y2 = self.encoder.block2_quant(y1)
        y2 = self.encoder.block2(y2)
        y2 = self.encoder.block2_dequant(y2)
        mu2, logvar2 = self.enc_head2(self.pool2(y2))

        y3 = self.encoder.block3_quant(y2)
        y3 = self.encoder.block3(y3)
        y3 = self.encoder.block3_dequant(y3)
        mu3, logvar3 = self.enc_head3(self.pool3(y3))

        #y4 = self.block4_quant(y3)
        #y4 = self.block4(y4)
        #y4 = self.block4_dequant(y4)
        return [mu0, mu1, mu2, mu3], [logvar0, logvar1, logvar2, logvar3]

    def forward_dec(self, z0, z1, z2, z3):
        y0 = self.dec_head0(z0)
        y0 = self.unpool0(y0)
        y0 = self.decoder.block0(y0)

        y1 = self.dec_head1(z1)
        y1 = self.unpool1(y1)
        y1 = self.decoder.block1(y1)
        y1 = self.decoder.block0(y1)
        
        y2 = self.dec_head2(z2)
        y2 = self.unpool2(y2)
        y2 = self.decoder.block2(y2)
        y2 = self.decoder.block1(y2)
        y2 = self.decoder.block0(y2)

        y3 = self.dec_head3(z3)
        y3 = self.unpool3(y3)
        y3 = self.decoder.block3(y3)
        y3 = self.decoder.block2(y3)
        y3 = self.decoder.block1(y3)
        y3 = self.decoder.block0(y3)

        return y0, y1, y2, y3

    def forward(self, x):
        mu, logvar = self.forward_enc(x)
        z = []
        for m, lvar in zip(mu, logvar):
            stdev = torch.exp(lvar / 2)
            eps = torch.randn_like(stdev)
            z.append(m + stdev * eps)
        return self.forward_dec(*z), mu, logvar

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, train_batch, batch_idx):
        x, _ = train_batch
        x_hat, mu, logvar = self.forward(x)
        head_weights = [1.0, 1.0, 1.0, 1.0]
        loss = torch.Tensor([0])
        loss = loss.to(self.device)
        for i,j,k,l in zip(x_hat, mu, logvar, head_weights):
            kl_loss = 0.5 * torch.sum(j.pow(2) + k.exp() - k - 1) / self.n_latent
            mse_loss = torch.nn.functional.mse_loss(i, x, reduction='mean')
            loss += l * (mse_loss + self.beta * kl_loss)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, _ = val_batch
        x_hat, mu, logvar = self.forward(x)
        head_weights = [0.125, 0.25, 0.5, 1.0]
        loss = torch.Tensor([0])
        loss = loss.to(self.device)
        for i,j,k,l in zip(x_hat, mu, logvar, head_weights):
            kl_loss = 0.5 * torch.sum(j.pow(2) + k.exp() - k - 1) / self.n_latent
            mse_loss = torch.nn.functional.mse_loss(i, x, reduction='mean')
            loss += l * (mse_loss + self.beta * kl_loss)
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
        mu, logvar = self.enc_head(x)
        kl_loss = []
        for m,lvar in zip(mu, logvar):
            kl_loss.append(0.5 * (m.pow(2) + lvar.exp() - lvar - 1))
        return kl_loss, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a VGGNet Early-Exit')
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
        help='Experiment Name'
    )
    parser.add_argument(
        '--n_layers',
        help='No. layers (11, 13, 16, or 19)'
    )
    parser.add_argument(
        '--n_latent',
        help='n_latent'
    )
    parser.add_argument(
        '--base_model',
        help='Fully trained base model'
    )
    args = parser.parse_args()
    model = VggnetVaeMultiHead(base_model=args.base_model, n_latent=int(args.n_latent), beta=1, learning_rate=1e-5)
    data = OodDataModule(args.train_dataset, args.val_dataset, None, None, batch_size=int(args.batch)) 
    trainer = pytorch_lightning.Trainer(
        accelerator='gpu',
        devices=1,
        deterministic=True,
        min_epochs=50,
        max_epochs=int(args.epochs),
        log_every_n_steps=10,
        logger=pytorch_lightning.loggers.TensorBoardLogger(save_dir='logs/'),
    )
    trainer.fit(model, datamodule=data)
    torch.save(model, f'{args.name}.pt')
