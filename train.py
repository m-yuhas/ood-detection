import argparse
import pytorch_lightning
import torch

from models.vggnet import VggnetWsvae, VggnetVae
from data.image_data import OodDataModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a VGGNet')
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
    args = parser.parse_args()
    model = VggnetWsvae(
        depth=int(args.n_layers),
        n_latent=int(args.n_latent),
        levels=5,
        alpha=1,
        beta=1,
        gamma=1,
        learning_rate=1e-5
    )
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
