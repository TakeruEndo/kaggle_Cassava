import sys
import os
import time
import random
from glob import glob
import warnings
from datetime import datetime
import joblib

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import GroupKFold, StratifiedKFold
from skimage import io
from sklearn.metrics import roc_auc_score, log_loss

import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

import timm
#from efficientnet_pytorch import EfficientNet
from scipy.ndimage.interpolation import zoom
from catalyst.data.sampler import BalanceClassSampler

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append('FMix-master')
from Model import efficientnet
from dataset import CassavaDataset
from transform import get_train_transforms, get_valid_transforms
from utils import seed_everything


def prepare_dataloader(cfg, df, trn_idx, val_idx, data_root='../input/cassava-leaf-disease-classification/train_images/'):

    train_ = df.loc[trn_idx, :].reset_index(drop=True)
    valid_ = df.loc[val_idx, :].reset_index(drop=True)

    train_ds = CassavaDataset(
        cfg, train_, data_root, transforms=get_train_transforms(cfg), output_label=True, one_hot_label=False, do_fmix=False, do_cutmix=False)
    valid_ds = CassavaDataset(
        cfg, valid_, data_root, transforms=get_valid_transforms(cfg), output_label=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.default.train_bs,
        pin_memory=False,
        drop_last=False,
        shuffle=True,
        num_workers=cfg.default.num_workers,
        # sampler=BalanceClassSampler(labels=train_['label'].values, mode="downsampling")
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=cfg.default.valid_bs,
        num_workers=cfg.default.num_workers,
        shuffle=False,
        pin_memory=False,
    )
    return train_loader, val_loader


def train_one_epoch(cfg, epoch, model, loss_fn, optimizer, train_loader, scaler, device, scheduler=None, schd_batch_update=False):
    model.train()

    running_loss = None

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        # print(image_labels.shape, exam_label.shape)
        with autocast():
            image_preds = model(imgs)  # output = model(input)
            # print(image_preds.shape, exam_pred.shape)

            loss = loss_fn(image_preds, image_labels)

            scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01

            if ((step + 1) % cfg.default.accum_iter == 0) or ((step + 1) == len(train_loader)):
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler is not None and schd_batch_update:
                    scheduler.step()

            if ((step + 1) % cfg.default.verbose_step == 0) or ((step + 1) == len(train_loader)):
                description = f'epoch {epoch} loss: {running_loss:.4f}'

                pbar.set_description(description)

    if scheduler is not None and not schd_batch_update:
        scheduler.step()


def valid_one_epoch(cfg, epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False):
    model.eval()

    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        image_preds = model(imgs)  # output = model(input)
        # print(image_preds.shape, exam_pred.shape)
        image_preds_all += [torch.argmax(image_preds,
                                         1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]

        loss = loss_fn(image_preds, image_labels)

        loss_sum += loss.item() * image_labels.shape[0]
        sample_num += image_labels.shape[0]

        if ((step + 1) % cfg.default.verbose_step == 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
            pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    print('validation multi-class accuracy = {:.4f}'.format(
        (image_preds_all == image_targets_all).mean()))

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum / sample_num)
        else:
            scheduler.step()


@hydra.main(config_name='config')
def main(cfg):
    train = pd.read_csv(cfg.common.train_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    seed_everything(cfg.default.seed)

    folds = StratifiedKFold(
        n_splits=cfg.default.fold_num, shuffle=True, random_state=cfg.default.seed).split(np.arange(train.shape[0]), train.label.values)

    for fold, (trn_idx, val_idx) in enumerate(folds):
        # we'll train fold 0 first
        if fold > 0:
            break

        print('Training with {} started'.format(fold))

        print(len(trn_idx), len(val_idx))
        train_loader, val_loader = prepare_dataloader(
            cfg, train, trn_idx, val_idx, data_root=cfg.common.img_path)

        model = efficientnet.CustomEfficientNet(
            cfg.default.model_arch, train.label.nunique(), pretrained=True).to(device)
        scaler = GradScaler()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.default.lr, weight_decay=cfg.default.weight_decay)

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=CFG['epochs']-1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.default.T_0, T_mult=1, eta_min=cfg.default.min_lr, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=25,
        #                                                max_lr=CFG['lr'], epochs=CFG['epochs'], steps_per_epoch=len(train_loader))
        # ---------------
        # SAMを使いたい
        # ---------------
        # base_optimizer = torch.optim.SGD
        # optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, lr=0.1, momentum=0.9, weight_decay=0.0005)
        # scheduler = None

        loss_tr = nn.CrossEntropyLoss().to(device)  # MyCrossEntropyLoss().to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)

        for epoch in range(cfg.default.epochs):
            train_one_epoch(cfg, epoch, model, loss_tr, optimizer, train_loader,
                            scaler, device, scheduler=scheduler, schd_batch_update=False)
            with torch.no_grad():
                valid_one_epoch(cfg, epoch, model, loss_fn, val_loader,
                                device, scheduler=None, schd_loss_update=False)
            torch.save(model.state_dict(),
                       f'{cfg.default.model_arch}_fold_{fold}_{epoch}')
        # torch.save(model.cnn_model.state_dict(),'{}/cnn_model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag']))
        del model, optimizer, train_loader, val_loader, scaler, scheduler
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
