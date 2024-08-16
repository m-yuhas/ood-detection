import torch
import pytorch_lightning


class Classifier(pytorch_lightning.LightningModule):

    def __init__(self, encoder, learning_rate=1e-5):
        super().__init__()
        self.learning_rate = learning_rate
        self.encoder = encoder
        self.smax = torch.nn.Softmax()

    def forward(self, x):
        y = self.encoder(x)
        return self.smax(y)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def predict_step(self, predict_batch, batch_idx):
        x, y = predict_batch
        y_hat = self(x)
        return y_hat, y
