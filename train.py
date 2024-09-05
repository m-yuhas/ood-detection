"""Train an OOD detector."""

import argparse
import random


import pytorch_lightning
import torch
import yaml


from models.vggnet import VggnetEnc, VggnetDec
from models.resnet import ResnetEnc, ResnetDec
from models.mobilenet import MobileNetEnc, MobileNetDec
from models.lenet import ConvNetEnc, ConvNetDec, LENET4
from lightning_modules.betavae import BetaVae
from data.image_data import OodDataModule
from data.coco_like_data import CocoLikeDataModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train an OOD detector')
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
        '--backbone',
        help='resnet18, resnet34'
    )
    parser.add_argument(
        '--n_latent',
        help='n_latent'
    )
    parser.add_argument(
        '--model-config',
        help='YAML file containing the model configuration for training.'
    )
    parser.add_argument(
        '--data-config',
        help='YAML file containing the dataset details.'
    )
    args = parser.parse_args()
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f.read())
    with open(args.data_config, 'r') as f:
        data_config = yaml.safe_load(f.read())
    if model_config['seed'] is not None:
        random.seed(model_config['seed'])
        pytorch_lightning.seed_everything(model_config['seed'], workers=True)
    encoder = None
    decoder = None
    backbone = model_config['backbone'].strip().lower()
    if backbone == 'mobilenet':
        encoder = MobileNetEnc(
            alpha=model_config['model hyperparameters']['alpha'],
            n_latent=2 * model_config['model hyperparameters']['n_latent'],
            input_shape=model_config['model hyperparameters']['input_dim'],
        )
        decoder = MobileNetDec(
            alpha=model_config['model hyperparameters']['alpha'],
            n_latent=model_config['model hyperparameters']['n_latent'],
            output_shape=model_config['model hyperparameters']['input_dim'],
        )
    elif backbone == 'resnet':
        encoder = ResnetEnc(**model_config['model hyperparamers'])
        decoder = ResnetDec(**model_config['model hyperparameters'])
    elif backbone =='vggnet':
        encoder = VggnetEnc(**model_config['model hyperparameters'])
        decoder = VggnetDec(**model_config['model hyperparameters'])
    elif backbone == 'lenet':
        encoder = ConvNetEnc(
            LENET4,
            model_config['model hyperparameters']['n_latent'] * 2,
            model_config['model hyperparameters']['input_dim']
        )
        decoder = ConvNetDec(
            LENET4,
            encoder.block_outputs,
            model_config['model hyperparameters']['n_latent'],
        )
    else:
        raise ValueError(f'Unsupported backbone: {backbone}')
        exit(2)
    assert encoder is not None and decoder is not None, "No backbone defined"
    model = None
    if model_config['detection model'] == 'betavae':
        model = BetaVae(
            encoder,
            decoder,
            model_config['training hyperparameters']['beta'],
            model_config['training hyperparameters']['learning rate'],
        )
    assert model is not None, "No detection model defined"
    data = None
    if data_config['dataset type'] == 'image':
        data = OodDataModule(
            data_config['train path'],
            data_config['val path'],
            data_config['test path'],
            resize=model_config['model hyperparameters']['input_dim'][1:],
            batch_size=model_config['training hyperparameters']['batch size']
        )
    elif data_config['dataset type'] == 'coco-like':
        data = CocoLikeDataModule(
            root=data_config['root'],
            train=data_config['train'],
            train_json=data_config['train_json'],
            val=data_config['val'],
            val_json=data_config['val_json'],
            test=data_config['test'],
            test_json=data_config['test_json'],
            shape=model_config['model hyperparameters']['input_dim'][1:],
            batch_size=model_config['training hyperparameters']['batch size']
        )
    checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor='val_loss',
        mode="min",
        filename='{epoch:02d}-{val_loss:.3f}',
    )
    trainer = pytorch_lightning.Trainer(
        accelerator='auto',
        callbacks=[checkpoint_callback],
        deterministic=True if model_config['seed'] is not None else False,
        devices='auto',
        max_epochs=model_config['training hyperparameters']['epochs'],
        min_epochs=model_config['training hyperparameters']['epochs'],
        logger=pytorch_lightning.loggers.TensorBoardLogger(save_dir='.'),
    )
    trainer.fit(model, datamodule=data)
    torch.save(model, f'{args.name}.pt')
