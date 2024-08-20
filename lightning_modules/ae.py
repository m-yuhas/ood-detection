import torch
import pytorch_lightning


class Autoencoder(pytorch_lightning.LightningModule):

    def __init__(self, encoder, decoder, learning_rate=1e-5):
        super().__init__()
        if encoder.n_latent != decoder.n_latent:
            raise ValueError('Latent dimensions of encoder and decoder must be equal')
        self.n_latent = encoder.n_latent
        self.learning_rate = learning_rate
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def reconstruction_loss(self, batch):
        x, _ = batch
        x_hat = self.forward(x)
        return torch.nn.functional.mse_loss(x_hat, x)

    def training_step(self, train_batch, batch_idx):
        loss = self.reconstruction_loss(train_batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self.reconstruction_loss(val_batch)
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        _, y = test_batch
        ood_score = self.reconstruction_loss(test_batch)
        return ood_score, y

    def predict_step(self, predict_batch, batch_idx):
        return self.test_step(predict_batch, batch_idx)
