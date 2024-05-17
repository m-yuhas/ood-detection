import argparse
import pytorch_lightning
import torch

from models.vggnet import VggnetWsvae, VggnetVae
from data.image_data import OodDataModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a VGGNet')
    parser.add_argument(
        '--test_dataset',
        help='Path to folder of training images'
    )
    parser.add_argument(
        '--batch',
        help='Batch size'
    )
    parser.add_argument(
        '--model_name',
        help='file name of *.pt model'
    )
    model = torch.load(args.model_name)
    data = OodDataModule(args.test_dataset, args.test_dataset, args.test_dataset, args.test_dataset, batch_size=int(args.batch)) 
    trainer = pytorch_lightning.Trainer(
        accelerator='gpu',
        devices=1,
        deterministic=True,
        min_epochs=1,
        max_epochs=1,
        logger=pytorch_lightning.loggers.TensorBoardLogger(save_dir='logs/'),
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=data)
    torch.save(model, f'{args.name}.pt')
