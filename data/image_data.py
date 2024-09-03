import pytorch_lightning
import torchvision
import torch

import os

class OodDataModule(pytorch_lightning.LightningDataModule):

    def __init__(self, train_path, val_path, test_path, resize, batch_size=32):
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self):
        train_set = torchvision.datasets.ImageFolder(
            root=self.hparams.train_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(self.hparams.resize, antialias=False)
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
                torchvision.transforms.Resize(self.hparams.resize, antialias=False)
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
                    torchvision.transforms.Resize(self.hparams.resize, antialias=False)
                ])
            ),
            batch_size=self.hparams.batch_size,
            num_workers=len(os.sched_getaffinity(0)),
            drop_last=True)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(
                root=self.hparams.test_path,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Resize(self.hparams.resize, antialias=False)
                ])
            ),
            batch_size=self.hparams.batch_size,
            num_workers=len(os.sched_getaffinity(0)),
            drop_last=True)
