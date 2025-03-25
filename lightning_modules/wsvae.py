import torch
import pytorch_lightning


class WeaklySupervisedVae(pytorch_lightning.LightningModule):

    def __init__(self, encoder, decoder, n_genfac, alpha=1,  beta=1, learning_rate=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.learning_rate = learning_rate
        self.encoder = encoder
        self.decoder = decoder
        self.n_genfac = n_genfac
        self.detection_head = torch.nn.Linear(self.encoder.n_latent // 2, n_genfac)

    def forward(self, x):
        z = self.encoder(x)
        mu, logvar = z[:, :z.shape[-1] // 2], z[:, z.shape[-1] // 2:]
        stdev = torch.exp(logvar / 2)
        eps = torch.randn_like(stdev)
        z = mu + stdev * eps
        y_hat = self.detection_head(mu)
        return self.decoder(z), mu, logvar, y_hat
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


    def wsloss(self, batch):
        x, y = batch
        y = torch.nn.functional.one_hot(y, self.n_genfac).float()
        x_hat, mu, logvar, y_hat = self.forward(x)
        kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1) / self.decoder.n_latent
        mse_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='mean')
        cls_loss = torch.nn.functional.cross_entropy(y_hat, y, reduction='mean')
        return mse_loss + kl_loss + self.alpha * cls_loss

    def training_step(self, train_batch, batch_idx):
        loss = self.wsloss(train_batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self.wsloss(val_batch)
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, _ = test_batch
        z = self.encoder(x)
        mu = z[:, :z.shape[-1] // 2]
        y_hat = self.detection_head(mu)
        return y_hat

    def predict_step(self, predict_batch, batch_idx):
        x, y = predict_batch
        z = self.encoder(x)
        mu = z[:, :z.shape[-1] // 2]
        y_hat = self.detection_head(mu)
        return y_hat, y
