import torch
import pytorch_lightning


class BetaVae(pytorch_lightning.LightningModule):

    def __init__(self, encoder, decoder, beta=1, learning_rate=1e-5):
        super().__init__()
        self.beta = beta
        self.n_latent = 500
        self.learning_rate = learning_rate
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        mu, logvar = z[:, :z.shape[-1] // 2], z[:, z.shape[-1] // 2:]
        stdev = torch.exp(logvar / 2)
        eps = torch.randn_like(stdev)
        z = mu + stdev * eps
        return self.decoder(z), mu, logvar
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


    def elbo_loss(self, batch):
        x, _ = batch
        x_hat, mu, logvar = self.forward(x)
        kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1) / self.n_latent
        mse_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='mean')
        return mse_loss + self.beta * kl_loss

    def training_step(self, train_batch, batch_idx):
        loss = self.elbo_loss(train_batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self.elbo_loss(val_batch)
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        loss = self.elbo_loss(test_batch)
        self.log('test_loss', kl_loss)
        return loss

    def predict_step(self, predict_batch, batch_idx):
        x, y = predict_batch
        _, mu, logvar = self.forward(x)
        kl_loss = 0.5 * mu.pow(2) + logvar.exp() - logvar - 1
        return kl_loss, y
