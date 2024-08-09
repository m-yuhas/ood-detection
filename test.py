import argparse
import pytorch_lightning
import torch
import numpy

from sklearn.metrics import roc_auc_score, precision_recall_curve

from lightning_modules.betavae import BetaVae
from models.vggnet import VggnetWsvae, VggnetVae
from models.resnet import Resnet18Enc, Resnet34Enc
from data.image_data import OodDataModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test an OOD Detector')
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
        accelerator='gpu' if torch.cuda.is_available else 'cpu',
        devices=1,
        deterministic=True,
        min_epochs=1,
        max_epochs=1,
        logger=pytorch_lightning.loggers.CSVLogger("logs", name=f'test_{args.model_name}'),
        enable_checkpointing=False,
    )
    num_imgs = len(data.val_dataloader().dataset)
    ood_score = numpy.zeros((num_imgs, 500))
    gt = numpy.zeros(num_imgs)
    results = trainer.predict(model, datamodule=data)
    for idx, r in enumerate(results):
        ood_score[idx * int(args.batch):(idx + 1) * int(args.batch), :] = r[0]
        gt[idx * int(args.batch):(idx + 1) * int(args.batch)] = r[1]
    gt[gt < 1] = 0
    gt[gt >= 1] = 1
    best_idx = 0
    best_auroc = 0
    for idx in range(500):
        auroc = roc_auc_score(gt, ood_score[:, idx])
        #print(f'AUROC: {auroc}')
        if auroc > best_auroc:
            best_idx = idx
            best_auroc = auroc
    print(f'Best AUROC: {best_auroc}')
    print(f'Latent Dimension: {best_idx}')
    prec, rec, thresh = precision_recall_curve(gt, ood_score[:, best_idx])
    thresh = thresh[rec[:-1] >= 0.8][-1]
    print(f'Thresh: {thresh}')
    P = ood_score[:, best_idx][gt >= 1]
    N = ood_score[:, best_idx][gt < 1]
    print(P.shape)
    print(N.shape)
    TP = P[P > thresh].size
    FP = P[P < thresh].size
    TN = N[N < thresh].size
    FN = N[N > thresh].size
    print(f'PPV@Rec=0.8: {TP/(TP + FP)}')
    print(f'NPV@Rec=0.8: {TN/(FN + TN)}')
