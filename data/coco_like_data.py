from typing import List, Tuple

from PIL import Image
import pytorch_lightning
import torchvision
import torch

import os
import json


class CocoLikeDataset(torch.utils.data.Dataset):

    def __init__(self, root: str, subdirs: List[str], json_f: str, transform: torch.nn.Module) -> None:
        super().__init__()
        self.transform = transform
        self.idx_to_class = {}
        img_list = []
        self.data = []
        with open(os.path.join(root, json_f), 'r') as f:
            img_list = json.loads(f.read())['images']
        for i, subdir in enumerate(subdirs):
            self.idx_to_class[i] = subdir
            for img in img_list:
                self.data.append((os.path.join(root, subdir, img['file_name']), i))

    def __getitem__(self, idx):
        #img = torchvision.io.read_image(self.data[idx][0])
        img = Image.open(self.data[idx][0])
        return self.transform(img), self.data[idx][1]

    def __len__(self):
        return len(self.data)


class CocoLikeDataModule(pytorch_lightning.LightningDataModule):

    def __init__(self, root: str, train: List[str], train_json: str, val: List[str], val_json: str, test: List[str], test_json: str, shape: Tuple[int], batch_size=32):
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self):
        train_set = CocoLikeDataset(
            root=self.hparams.root,
            subdirs=self.hparams.train,
            json_f=self.hparams.train_json,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(self.hparams.shape, antialias=False)
            ])
        )
        return torch.utils.data.DataLoader(
            train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=len(os.sched_getaffinity(0)),
            drop_last=True)

    def val_dataloader(self):
        val_set = CocoLikeDataset(
            root=self.hparams.root,
            subdirs=self.hparams.val,
            json_f=self.hparams.val_json,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(self.hparams.shape, antialias=False)
            ])
        )
        return torch.utils.data.DataLoader(
            val_set,
            batch_size=min(len(val_set), self.hparams.batch_size),
            num_workers=len(os.sched_getaffinity(0)),
            drop_last=True)

    def test_dataloader(self):
        return CocoLikeDataset(
            torchvision.datasets.ImageFolder(
                root=self.hparams.root,
                subdirs=self.hparams.test,
                json_f=self.hparams.test_json,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Resize(self.hparams.shape, antialias=False)
                ])
            ),
            batch_size=self.hparams.batch_size,
            num_workers=len(os.sched_getaffinity(0)),
            drop_last=True)

    def predict_dataloader(self):
        return CocoLikeDataset(
            torchvision.datasets.ImageFolder(
                root=self.hparams.root,
                subdirs=self.hparams.test,
                json_f=self.hparams.test_json,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Resize(self.hparams.shape, antialias=False)
                ])
            ),
            batch_size=self.hparams.batch_size,
            num_workers=len(os.sched_getaffinity(0)),
            drop_last=True)
