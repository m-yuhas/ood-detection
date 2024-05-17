import argparse
import pytorch_lightning
import torch
import numpy

from sklearn.metrics import roc_auc_score

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
    args = parser.parse_args()
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
    num_imgs = len(data.val_dataloader().dataset)
    ood_score = numpy.zeros(num_imgs)
    gt = numpy.zeros(num_imgs)
    results = trainer.predict(model, datamodule=data)
    for idx, r in enumerate(results):
        ood_score[idx * int(args.batch):(idx + 1) * int(args.batch)] = r[0]
        gt[idx * int(args.batch):(idx + 1) * int(args.batch)] = r[1]
    gt[gt <= 4] = 0
    gt[gt > 4] = 1
    auroc = roc_auc_score(gt, ood_score)
    print(f'AUROC: {auroc}')