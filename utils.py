import os
import sys
import random
from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler

import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

from model.nn_model import *
from loss import *


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def init_logger(log_file):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def select_model(model_name, n_class):
    if model_name == 'deit_base_patch16_224':
        model = CustomDeiT(model_name, n_class, pretrained=False)
    elif model_name == 'vit_base_patch16_384':
        model = CustomViT(model_name, n_class, pretrained=False)
    elif model_name == 'resnext50_32x4d':
        model = CustomResNext(model_name, n_class, pretrained=True)
    elif model_name == 'tf_efficientnet_b4_ns':
        model = CustomEfficientNet(model_name, n_class, pretrained=True)
    elif model_name == 'tf_efficientnet_b5_ns':
        model = CustomEfficientNet(model_name, n_class, pretrained=True)
    elif model_name == 'tf_efficientnet_b6_ns':
        model = CustomEfficientNet(model_name, n_class, pretrained=True)
    elif model_name == 'tf_mixnet_s':
        model = CustomEfficientNet(model_name, n_class, pretrained=True)
    else:
        print('Model arch is not correct')
        sys.exit()
    return model


def select_loss(loss_name):
    if loss_name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif loss_name == 'FocalCosineLoss':
        return FocalCosineLoss()
    elif loss_name == 'SymmetricCrossEntropy':
        return SymmetricCrossEntropy()
    elif loss_name == 'TaylorCrossEntropyLoss':
        return TaylorCrossEntropyLoss()
    else:
        print('Loss name is incorrect')
        sys.exit()


def get_scheduler(cfg, optimizer):
    if cfg.shd_para.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=cfg.shd_para.factor, patience=cfg.shd_para.patience, verbose=True, eps=cfg.shd_para.eps)
    elif cfg.shd_para.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.shd_para.T_max, eta_min=cfg.shd_para.min_lr, last_epoch=-1)
    elif cfg.shd_para.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.shd_para.T_0, T_mult=1, eta_min=cfg.shd_para.min_lr, last_epoch=-1)
    else:
        print('scheduler name is not collect')
        sys.exit()
    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
