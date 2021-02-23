import logging
import pandas as pd
import os, glob, time, datetime, sys
import copy
import shutil
import argparse
from pytz import timezone, utc

import random
import torch
# import torch.distributed as dist
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from data import *


# sys.path.append('pytorch-grad-cam')
# sys.path.append('A-journey-into-Convolutional-Neural-Network-visualization-')
# import gradcam #GradCam


class Accumulator:
    def __init__(self):
        self.metrics = defaultdict(lambda: 0.)

    def add(self, key, value):
        self.metrics[key] += value

    def add_dict(self, dict):
        for key, value in dict.items():
            self.add(key, value)

    def __getitem__(self, item):
        return self.metrics[item]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def get_dict(self):
        return copy.deepcopy(dict(self.metrics))

    def items(self):
        return self.metrics.items()

    def __str__(self):
        return str(dict(self.metrics))

    def __truediv__(self, other):
        newone = Accumulator()
        for key, value in self.items():
            if isinstance(other, str):
                if other != key:
                    newone[key] = value / self[other]
                else:
                    newone[key] = value
            else:
                newone[key] = value / other
        return newone


# # ref : https://github.com/CoinCheung/pytorch-loss        
class FocalLossV2(nn.Module):
    '''
    This use better formula to compute the gradient, which has better numeric stability
    '''
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean'):
        super(FocalLossV2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, label):
        loss = FocalSigmoidLossFuncV2.apply(logits, label, self.alpha, self.gamma)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

class FocalSigmoidLossFuncV2(torch.autograd.Function):
    '''
    compute backward directly for better numeric stability
    '''
    @staticmethod
    def forward(ctx, logits, label, alpha, gamma):
        logits = logits.float()
        coeff = torch.empty_like(logits).fill_(1 - alpha)
        coeff[label == 1] = alpha

        probs = torch.sigmoid(logits)
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        probs_gamma = probs ** gamma
        probs_1_gamma = (1. - probs) ** gamma

        ctx.coeff = coeff
        ctx.probs = probs
        ctx.log_probs = log_probs
        ctx.log_1_probs = log_1_probs
        ctx.probs_gamma = probs_gamma
        ctx.probs_1_gamma = probs_1_gamma
        ctx.label = label
        ctx.gamma = gamma

        term1 = probs_1_gamma * log_probs
        term2 = probs_gamma * log_1_probs
        loss = torch.where(label == 1, term1, term2).mul_(coeff).neg_()
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        '''
        compute gradient of focal loss
        '''
        coeff = ctx.coeff
        probs = ctx.probs
        log_probs = ctx.log_probs
        log_1_probs = ctx.log_1_probs
        probs_gamma = ctx.probs_gamma
        probs_1_gamma = ctx.probs_1_gamma
        label = ctx.label
        gamma = ctx.gamma

        term1 = (1. - probs - gamma * probs * log_probs).mul_(probs_1_gamma).neg_()
        term2 = (probs - gamma * (1. - probs) * log_1_probs).mul_(probs_gamma)

        grads = torch.where(label == 1, term1, term2).mul_(coeff).mul_(grad_output)
        return grads, None, None, None

from torch.optim.lr_scheduler import _LRScheduler
from math import pi, cos

class CosineWarmupLR(_LRScheduler):
    '''
    Cosine lr decay function with warmup.
    Ref: https://github.com/PistonY/torch-toolbox/blob/master/torchtoolbox/optimizer/lr_scheduler.py
         https://github.com/Randl/MobileNetV3-pytorch/blob/master/cosine_with_warmup.py
    Lr warmup is proposed by
        `Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour`
        `https://arxiv.org/pdf/1706.02677.pdf`
    Cosine decay is proposed by
        `Stochastic Gradient Descent with Warm Restarts`
        `https://arxiv.org/abs/1608.03983`
    Args:
        optimizer (Optimizer): optimizer of a model.
        iter_in_one_epoch (int): number of iterations in one epoch.
        epochs (int): number of epochs to train.
        lr_min (float): minimum(final) lr.
        warmup_epochs (int): warmup epochs before cosine decay.
        last_epoch (int): init iteration. In truth, this is last_iter
    Attributes:
        niters (int): number of iterations of all epochs.
        warmup_iters (int): number of iterations of all warmup epochs.
        cosine_iters (int): number of iterations of all cosine epochs.
    '''

    def __init__(self, optimizer, epochs, iter_in_one_epoch, lr_min=0, warmup_epochs=0, last_epoch=-1):
        self.lr_min = lr_min
        self.niters = epochs * iter_in_one_epoch
        self.warmup_iters = iter_in_one_epoch * warmup_epochs
        self.cosine_iters = iter_in_one_epoch * (epochs - warmup_epochs)
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            return [(self.lr_min + (base_lr - self.lr_min) * self.last_epoch / self.warmup_iters) for base_lr in
                    self.base_lrs]
        else:
            return [(self.lr_min + (base_lr - self.lr_min) * (
                    1 + cos(pi * (self.last_epoch - self.warmup_iters) / self.cosine_iters)) / 2) for base_lr in
                    self.base_lrs]

def run_epoch(model, dataloader, criterion, optimizer=None, epoch=0, scheduler=None, device='cpu'):
    from pytorch_lightning.metrics.functional import confusion_matrix
    from pytorch_lightning.metrics import Precision, Accuracy, Recall
    from sklearn.metrics import roc_auc_score, average_precision_score

    metrics = Accumulator()
    cnt = 0
    total_steps = len(dataloader)
    steps = 0
    running_corrects = 0
    

    accuracy = Accuracy()
    precision = Precision(num_classes=2)
    recall = Recall(num_classes=2)

    preds_epoch = []
    labels_epoch = []
    for inputs, labels in dataloader:
        steps += 1

        inputs = inputs.to(device) # torch.Size([2, 1, 224, 224])
        labels = labels.to(device).unsqueeze(1).float() ## torch.Size([2, 1])

        outputs = model(inputs) # [batch_size, nb_classes]
        # logger.info(outputs.size()) # tmp
        # logger.info(outputs) # tmp

        loss = criterion(outputs, labels)

        if optimizer:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        preds_epoch.extend(torch.sigmoid(outputs).tolist())
        labels_epoch.extend(labels.tolist())

        logger.info('predicted probabilities : ')
        logger.info(torch.sigmoid(outputs).tolist())
        logger.info('labels (groud truth): ')
        logger.info(labels.tolist())

        pred_decision = (torch.sigmoid(outputs)>0.5).long()
        
        conf = torch.flatten(confusion_matrix(pred_decision, labels, num_classes=2))
        tn, fp, fn, tp = conf

        metrics.add_dict({
            'data_count': len(inputs),
            'loss': loss.item() * len(inputs),
            'tp': tp.item(),
            'tn': tn.item(),
            'fp': fp.item(),
            'fn': fn.item(),
        })
        cnt += len(inputs)

        if scheduler:
            scheduler.step()
        del outputs, loss, inputs, labels, pred_decision
    logger.info(f'cnt = {cnt}')

    metrics['loss'] /= cnt

    def safe_div(x,y):
        if y == 0:
            return 0
        return x / y
    _TP,_TN, _FP, _FN = metrics['tp'], metrics['tn'], metrics['fp'], metrics['fn']
    acc = (_TP+_TN)/cnt
    sen = safe_div(_TP , (_TP + _FN))
    spe = safe_div(_TN , (_FP + _TN))
    prec = safe_div(_TP , (_TP + _FP))

    metrics.add('accuracy', acc)
    metrics.add('sensitivity', sen)
    metrics.add('specificity', spe)
    metrics.add('precision', prec)
    try:
        auc = roc_auc_score(labels_epoch, preds_epoch)
        aupr = average_precision_score(labels_epoch, preds_epoch)
        metrics.add('auroc', auc)
        metrics.add('aupr', aupr)
    except ValueError :
        logger.warn('fail to calculate auc, aupr')
    logger.info(metrics)

    return metrics, preds_epoch, labels_epoch

def train_and_eval(args, fold_num, eff_net='b0', max_epoch=10, save_path='.', batch_size=8, multigpu=True): #TODO
    os.makedirs(save_path, exist_ok=True) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'DEVICE: {device}')
    logger.info('====='*10)
    logger.info(' '*20 + f'FOLD: {fold_num} / 10')
    logger.info('====='*10)

    test_dataset, trainloader, validloader, testloader = get_dataloaders(args, ImageDataset, batch=batch_size, root='.', fold_num=fold_num, multinode=multigpu, augment=False)
    
    if args.criterion =='FocalLoss':
        criterion = FocalLossV2()
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    rs = dict()

    if args.trained_model is not None :
        # perform only evaluation
        model = get_trained_model(args.trained_model)
        model.to(device)

    else: 
        # pretrained_online = True
        # if pretrained_online :

        # load pretrained model and retrain using our data
        model = EfficientNet.from_pretrained(f'efficientnet-{eff_net}', num_classes=1, weights_path='./efficientnet-b3-5fb5a3c3.pth',
            in_channels=1 if disease=='Sinusitis' else 3)
        model = model.to(device)

        # else :
        #     # load trained parameter on the local disk
        #     state_dict = torch.load('./efficientnet-b3-5fb5a3c3.pth', map_location="cuda:0")
        #     state_dict.pop('_fc.weight')
        #     state_dict.pop('_fc.bias')

        #     model = EfficientNet.from_name(f'efficientnet-{eff_net}', num_classes=1)
        #     model.load_state_dict(state_dict, strict=False)           
        #     model = model.to(device)
        # logger.info(model.parameters()) #tmp


        if multigpu :
            model = nn.DataParallel(model)

        if args.optimizer =='Adam':
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        else: 
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) #weight_decay=1e-5

        logger.info(f'optimizer: {optimizer}')
        logger.info(f'criterion: {criterion}')

        if args.scheduler =='lr_scheduler':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        else:
            scheduler = CosineWarmupLR(optimizer=optimizer, epochs=max_epoch, iter_in_one_epoch=len(trainloader), lr_min=0.0001,
                                warmup_epochs=0)

        logger.info(f'scheduler: {scheduler}')

        since = time.time()


        for epoch in range(max_epoch):
            logger.info('-' * 10)
            logger.info('Start Epoch {}/{}'.format(epoch, max_epoch - 1))

            logger.info('====='*10)
            logger.info(' '*20 + f'START EPOCH {epoch}')
            logger.info('====='*10)

            model.train()
            logger.info('-----'*10)
            logger.info(' '*20 + f'TRAINING')
            logger.info('-----'*10)
            rs['train'],_ ,_ = run_epoch(model, trainloader, criterion, optimizer=optimizer, scheduler=scheduler, device=device)

            logger.info('-----'*10)
            logger.info(' '*20 + f'VALIDATION')
            logger.info('-----'*10)

            model.eval()
            with torch.no_grad():
                rs['valid'],preds_valid, labels_valid = run_epoch(model, validloader, criterion, optimizer=None, device=device)

            logger.info(
                f'[ EPOCH {epoch} ]'
                f'[ TRAIN ] loss={rs["train"]["loss"]:.4f}, accuracy={rs["train"]["accuracy"]:.4f} '
                f'[ VALID ] loss={rs["valid"]["loss"]:.4f}, accuracy={rs["valid"]["accuracy"]:.4f}, precision={rs["valid"]["precision"]:.4f}, recall={rs["valid"]["sensitivity"]:.4f}'
            )
            # tensorboard
            writer.add_scalars(f"Accuracy/fold{fold_num}",{'train':rs["train"]["accuracy"], 'valid':rs["valid"]["accuracy"]}, epoch)
            writer.add_scalars(f"Precision/fold{fold_num}",{'train':rs["train"]["precision"],'valid':rs["valid"]["precision"]}, epoch)
            writer.add_scalars(f"Specificity/fold{fold_num}",{'train':rs["train"]["specificity"],'valid':rs["valid"]["specificity"]}, epoch)
            writer.add_scalars(f"Sensitivity/fold{fold_num}",{'train':rs["train"]["sensitivity"],'valid':rs["valid"]["sensitivity"]}, epoch)
            writer.add_scalars(f"AUPR/fold{fold_num}",{'train':rs["train"]["aupr"], 'valid':rs["valid"]["aupr"]}, epoch)
            writer.add_scalars(f"AUROC/fold{fold_num}",{'train':rs["train"]["auroc"], 'valid':rs["valid"]["auroc"]}, epoch)
            writer.add_scalars(f"Loss/fold{fold_num}",{'train':rs["train"]["loss"], 'valid':rs["valid"]["loss"]}, epoch)
            writer.flush()
        time_elapsed = time.time() - since
        logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    logger.info('====='*10)
    logger.info(' '*20 + f'EVALUATION')
    logger.info('====='*10)
    model.eval()
    with torch.no_grad():
        rs['test'], preds_test, labels_test = run_epoch(model, testloader, criterion, optimizer=None, device=device)

    if args.trained_model is None :
        os.makedirs(f'{save_path}/weights/', exist_ok=True)
        torch.save({
                            'epoch': epoch,
                            'log': {
                                'train': rs['train'].get_dict(),
                                'valid': rs['valid'].get_dict(),
                                'test': rs['test'].get_dict(),
                            },
                            'optimizer': optimizer.state_dict(),
                            'model': model.state_dict(),
                        }, f'{save_path}/weights/model_weights_{eff_net}_fold{fold_num}.pth')

    logger.info(f'[ TEST ] loss={rs["test"]["loss"]:.3f}, accuracy={rs["test"]["accuracy"]:.3f}, AUROC={rs["test"]["auroc"]:.3f}, sensitivity={rs["test"]["sensitivity"]:.3f}')
    writer.add_scalars(f"Accuracy/fold{fold_num}",{f'test':rs["test"]["accuracy"] },max_epoch)
    writer.add_scalars(f"Precision/fold{fold_num}",{f'test':rs["test"]["precision"] },max_epoch)
    writer.add_scalars(f"Specificity/fold{fold_num}",{f'test':rs["test"]["specificity"] },max_epoch)
    writer.add_scalars(f"Sensitivity/fold{fold_num}",{f'test':rs["test"]["sensitivity"] },max_epoch)
    writer.add_scalars(f"AUPR/fold{fold_num}",{f'test':rs["test"]["aupr"] },max_epoch)
    writer.add_scalars(f"AUROC/fold{fold_num}",{f'test':rs["test"]["auroc"] },max_epoch)
    writer.add_scalars(f"Loss/fold{fold_num}",{f'test':rs["test"]["loss"] },max_epoch)
    writer.flush()

    # record final evaluation metric
    savedir = f'{save_path}/metrics'
    os.makedirs(savedir,exist_ok=True) 
    
    def save_metric_csv(x):
        dic = rs[x].get_dict()
        pd.DataFrame.from_dict(dic, orient='index').to_csv(os.path.join(savedir, f"metric_fold{fold_num}_")+f'{x}.csv', header=False)
    save_metric_csv('test')

    preds_test_df = pd.DataFrame({'pred':preds_test, 'label':labels_test })
    preds_test_df.to_csv(os.path.join(savedir, f"testset_prediction_fold{fold_num}.csv"),index=False)
    if args.trained_model is None:
        save_metric_csv('valid')

    return None


def get_trained_model(model_path):
    if model_path == './efficientnet-b3-5fb5a3c3.pth':
        model = EfficientNet.from_pretrained(f'efficientnet-b3', num_classes=1, weights_path='./efficientnet-b3-5fb5a3c3.pth',
            in_channels=3)
    else:
        # load trained parameter
        ch = torch.load(model_path)
        state_dict = ch['model'] # dict_keys(['epoch', 'log', 'optimizer', 'model'])

        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v

        model = EfficientNet.from_name('efficientnet-b3', num_classes=1, in_channels=3)
        model.load_state_dict(new_state_dict, strict=False)

    model.eval()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=None,
                        help='Index of 10 fold cross-validation to train model')
    parser.add_argument('--trained_model', type=str, default=None,
                        help='Load a trained model in the path and perform only evaluation, without training process')

    parser.add_argument('--criterion', type=str, default='BCE',
                        help='loss fuction. default "BCE". otherwise, "FocalLoss" ')

    parser.add_argument('--optimizer', type=str, default="SGD",
                        help='optimizer. default "SGD". otherwise, "Adam"')

    parser.add_argument('--scheduler', type=str, default="CosineWarmupLR",
                        help='learning rate scheduler. default "CosineWarmupLR". otherwise, "lr_scheduler"')
    # parser.add_argument('--tag', type=str, default='')

    args = parser.parse_args()
    # if args.tag =='':
    #     today_datetime = datetime.datetime.now().strftime("%y%m%d_%h%m")
    # else:
    #     today_datetime = args.tag 

    disease = ['Sinusitis','Oral_cancer'][1]
    eff_model = ['b0','b3'][1]
    KST = timezone('Asia/Seoul')
    today_datetime = datetime.datetime.utcnow().strftime("%y%m%d_%Hh%Mm")    

    logger = get_logger(f'{disease}_EfficientNet_{today_datetime}', resetlogfile=True)
    logger.setLevel(logging.INFO)
    logger.info(args)
    tb = f'./log/{today_datetime}/tensorboard'
    if os.path.exists(tb):
        shutil.rmtree(tb, ignore_errors=True)
    os.makedirs(tb, exist_ok=True)
    writer = SummaryWriter(tb)

    if args.fold is None:
        cv_fold = range(10)
    else:
        assert isinstance(args.fold,int)
        cv_fold = [args.fold]
    # print evironment
    # logger.info('------ ENVIRONMENTS -----')
    # logger.info('* CPU :')
    #logger.info(os.system("cat /proc/cpuinfo | grep 'model name' | uniq" ))
    # logger.info(os.system("lscpu | head -n 13"))
    # logger.info('* GPU :')
    # logger.info(os.system("nvidia-smi"))
    #logger.info('* Count of GPUs  : ', torch.cuda.device_count())
    #logger.info('* Available GPUs : ',torch.cuda.is_available())
    # logger.info("* RAM (GB):")
    # logger.info(os.system("totalm=$(free -g | awk '/^Mem:/{print $2}') ; echo $totalm"))
    # logger.info('* HDD :')
    # logger.info(os.system("lsblk"))
    # logger.info("* OS : ")
    # logger.info(os.system("lsb_release -a"))
    # logger.info('* Python version :')
    # logger.info(os.system('python -V'))

    # logger.info('* PyTorch version :')
    # logger.info( torch.__version__)
        
    np.random.seed(0)
    torch.manual_seed(0)

    since = time.time()
    logger.info(since)
    save_validationlist()

    _batch_size = 32#512 #256 
    if args.trained_model:
        _batch_size = 16
    _max_epoch = 10
    logger.info(f'batch_size : {_batch_size}, max_epoch : {_max_epoch}')

    # for ii in [0]:
    for ii in cv_fold:
        _ = train_and_eval(args, fold_num=ii, eff_net=eff_model, max_epoch=_max_epoch, batch_size=_batch_size, multigpu=True, save_path=f'./log/{today_datetime}')

    writer.close()
    time_elapsed = time.time() - since
    logger.info('Complete in {:.0f}m {:.0f}s !!'.format(time_elapsed // 60, time_elapsed % 60))
