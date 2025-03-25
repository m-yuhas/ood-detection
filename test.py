import argparse
import pytorch_lightning
import torch
import torchvision
import numpy
import yaml

from sklearn.metrics import roc_auc_score, precision_recall_curve

from lightning_modules.betavae import BetaVae
from models.resnet import ResnetEnc, ResnetDec
from models.mobilenet import MobileNetEnc, MobileNetDec
from data.image_data import OodDataModule
from data.coco_like_data import CocoLikeDataModule


class OodDetector(pytorch_lightning.LightningModule):
    def __init__(self, weights: torch.nn.Module, size: int, latent: int, thresh: float):
        super().__init__()
        self.resize = torchvision.transforms.Resize((size, size))
        self.stem = weights.encoder
        self.thresh = thresh
        self.latent = latent

    def forward(self, x):
        x = self.resize(x)
        z = self.stem(x)
        mu, logvar = z[:, :z.shape[-1] // 2], z[:, z.shape[-1] // 2:]
        z = 0.5 * mu[0, self.latent].pow(2) + logvar[0, self.latent].exp() - logvar[0, self.latent] - 1
        if z.item() > self.thresh:
            return torch.ones(1)
        else:
            return torch.zeros(1)

class QuantizedModel(pytorch_lightning.LightningModule):
    def __init__(self, weights: torch.nn.Module):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.model = weights.encoder
        self.dequant_mu = torch.ao.quantization.DeQuantStub()
        self.dequant_logvar = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        #x = x.cpu()
        x = self.quant(x)
        z = self.model(x)
        mu, logvar = z[:, :z.shape[-1] // 2], z[:, z.shape[-1] // 2:]
        logvar = self.dequant_logvar(logvar)
        mu = self.dequant_mu(mu)
        return mu, logvar


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test an OOD Detector')
    parser.add_argument(
        '--data-config',
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
    parser.add_argument(
        '--n_latent',
        help='number of latent dimensions'
    )
    parser.add_argument(
        '--size',
        type=int,
        help='dimensions of image',
    )
    parser.add_argument(
        '--output_name',
        help='torchscript output name'
    )
    args = parser.parse_args()
    model = torch.load(args.model_name)
    #data = OodDataModule(args.test_dataset, args.test_dataset, args.test_dataset, (128, 128), batch_size=int(args.batch))
    with open(args.data_config, 'r') as f:
        data_config = yaml.safe_load(f.read()) 
    data = CocoLikeDataModule(
        root=data_config['root'],
        train=data_config['train'],
        train_json=data_config['train_json'],
        val=data_config['val'],
        val_json=data_config['val_json'],
        test=data_config['test'],
        test_json=data_config['test_json'],
        shape=(args.size, args.size),
        batch_size=int(args.batch)
    ) 
    trainer = pytorch_lightning.Trainer(
        accelerator='gpu' if torch.cuda.is_available else 'cpu',
        devices=1,
        deterministic=True,
        min_epochs=1,
        max_epochs=1,
        logger=pytorch_lightning.loggers.CSVLogger("logs", name=f'test_{args.model_name}'),
        enable_checkpointing=False,
    )
    num_imgs = len(data.predict_dataloader().dataset)
    ood_score = numpy.zeros((num_imgs, int(args.n_latent)))
    gt = numpy.zeros(num_imgs)
    results = trainer.predict(model, datamodule=data)
    for idx, r in enumerate(results):
        ood_score[idx * int(args.batch):(idx + 1) * int(args.batch), :] = r[0]
        gt[idx * int(args.batch):(idx + 1) * int(args.batch)] = r[1]
    gt[gt < 1] = 0
    gt[gt >= 1] = 1
    best_idx = 0
    best_auroc = 0
    for idx in range(int(args.n_latent)):
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
    TP = P[P > thresh].size
    FP = P[P < thresh].size
    TN = N[N < thresh].size
    FN = N[N > thresh].size
    print(f'SENS@Rec=0.8: {TP/(TP + FN)}')
    print(f'SPEC@Rec=0.8: {TN/(FP + TN)}')
    print(f'CONFUSION MATRIX:')
    print(f'      PREDICTED')
    print(f'       T     F    ')
    print(f'    +------+------+ ')
    print(f'   T| {TP} | {FN} | ')
    print(f'GT  +------+------+ ')
    print(f'   F| {FP} | {TN} | ')
    print(f'    +------+------+ ')

    TT = 0
    TF = 0
    FT = 0
    FF = 0
    p_nprev = numpy.zeros(P.size - 1)
    p_n = numpy.zeros(P.size - 1)
    for i in range(1, P.size):
        p_nprev[i-1] = P[i-1]
        p_n[i-1] = P[i]
        if P[i] >= thresh:
            if P[i-1] >= thresh:
                TT += 1
            else:
                FT += 1
        else:
            if P[i-1] >= thresh:
                TF += 1
            else:
                FF += 1

    n_nprev = numpy.zeros(P.size - 1)
    n_n = numpy.zeros(N.size - 1)
    for i in range(1, N.size):
        n_nprev[i-1] = N[i-1]
        n_n[i-1] = N[i]
        if N[i] >= thresh:
            if N[i-2] >= thresh:
                TT += 1
            else:
                FT += 1
        else:
            if N[i-1] >= thresh:
                TF += 1
            else:
                FF += 1
    print(f'CORRELATION MATRIX:')
    print(f'      TIME N')
    print(f'       T     F    ')
    print(f'     +------+------+ ')
    print(f'    T| {TT} | {TF} | ')
    print(f'N-1  +------+------+ ')
    print(f'    F| {FT} | {FF} | ')
    print(f'     +------+------+ ')


    print(numpy.corrcoef(p_nprev,p_n))
    print(numpy.corrcoef(n_nprev,n_n))
    prev = numpy.concatenate((p_nprev, n_nprev), axis=0)
    cur = numpy.concatenate((p_n, n_n), axis=0)
    print(numpy.corrcoef(prev,cur))


    m = OodDetector(model, args.size, best_idx, thresh).to(torch.device('cuda'))
    m = m.eval()
    x = torch.randn(1,3,640,480).to(torch.device('cuda'))
    m(x)
    m.to_torchscript(args.output_name, method='trace', example_inputs=(x))


    model_fp32 = QuantizedModel(model)

    # model must be set to eval mode for static quantization logic to work
    model_fp32.eval()

    # attach a global qconfig, which contains information about what kind
    # of observers to attach. Use 'x86' for server inference and 'qnnpack'
    # for mobile inference. Other quantization configurations such as selecting
    # symmetric or asymmetric quantization and MinMax or L2Norm calibration techniques
    # can be specified here.
    # Note: the old 'fbgemm' is still available but 'x86' is the recommended default
    # for server inference.
    # model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')

    # Fuse the activations to preceding layers, where applicable.
    # This needs to be done manually depending on the model architecture.
    # Common fusions include `conv + relu` and `conv + batchnorm + relu`
    model_fp32_fused = torch.ao.quantization.fuse_modules(
        model_fp32,
        [
            ['model.block0.conv', 'model.block0.bn', 'model.block0.act']
        ]
    )

    # Prepare the model for static quantization. This inserts observers in
    # the model that will observe activation tensors during calibration.
    model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)
    model_fp32_prepared = model_fp32_prepared.cpu()
    # calibrate the prepared model to determine quantization parameters for activations
    # in a real world setting, the calibration would be done with a representative dataset

    #for img in data.predict_dataloader():
    #    x, y = img
    #    model_fp32_prepared(x.cpu())

    # Convert the observed model to a quantized model. This does several things:
    # quantizes the weights, computes and stores the scale and bias value to be
    # used with each activation tensor, and replaces key operators with quantized
    # implementations.
    model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

    # run the model, relevant calculations will happen in int8
    #trainer = pytorch_lightning.Trainer(
    #    accelerator='cpu',
    #    devices=1,
    #    deterministic=True,
    #    min_epochs=1,
    #    max_epochs=1,
    #    logger=pytorch_lightning.loggers.CSVLogger("logs", name=f'test_{args.model_name}'),
    #    enable_checkpointing=False,
    #)
    num_imgs = len(data.predict_dataloader().dataset)
    ood_score = numpy.zeros((num_imgs, int(args.n_latent)))
    gt = numpy.zeros(num_imgs)
    model_int8 = model_int8.cpu()
    #results = trainer.predict(model_int8, datamodule=data)
    model_int8 = model.to(torch.float8_e4m3fn)
    for idx, img in enumerate(data.predict_dataloader()):
        x, y = img
        r = model_int8(x.cpu())
        ood_score[idx * int(args.batch):(idx + 1) * int(args.batch), :] = r[0]
        gt[idx * int(args.batch):(idx + 1) * int(args.batch)] = r[1]
    gt[gt < 1] = 0
    gt[gt >= 1] = 1
    best_idx = 0
    best_auroc = 0
    for idx in range(int(args.n_latent)):
        auroc = roc_auc_score(gt, ood_score[:, idx])
        #print(f'AUROC: {auroc}')
        if auroc > best_auroc:
            best_idx = idx
            best_auroc = auroc
    print(f'Best AUROC (QINT8): {best_auroc}')
    print(f'Latent Dimension (QINT8): {best_idx}')
    prec, rec, thresh = precision_recall_curve(gt, ood_score[:, best_idx])
    thresh = thresh[rec[:-1] >= 0.8][-1]
    print(f'Thresh (QINT8): {thresh}')
    
