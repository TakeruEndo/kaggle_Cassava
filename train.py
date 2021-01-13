import sys
import os
import time
import random
from glob import glob
import warnings
import logging
warnings.simplefilter('ignore')
from datetime import datetime
import joblib
import shutil

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
from torch.utils.tensorboard import SummaryWriter

import timm
#from efficientnet_pytorch import EfficientNet
from scipy.ndimage.interpolation import zoom
from catalyst.data.sampler import BalanceClassSampler

import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import CassavaDataset
from transform import get_train_transforms, get_valid_transforms
from utils import seed_everything, init_logger, select_model, AverageMeter, get_scheduler, get_score


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


def train_fn(cfg, epoch, model, loss_fn, optimizer, train_loader, scaler, device, writer, scheduler=None, schd_batch_update=False):
    model.train()

    running_loss = None
    losses = AverageMeter()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        if epoch == 0 and step == 0:
            grid = torchvision.utils.make_grid(imgs)
            writer.add_image('train_images', grid, 0)

        with autocast():
            y_preds = model(imgs)  # output = model(input)

            loss = loss_fn(y_preds, image_labels)
            losses.update(loss.item(), cfg.default.train_bs)
            scaler.scale(loss).backward()

            if ((step + 1) % cfg.default.accum_iter == 0) or ((step + 1) == len(train_loader)):
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None and schd_batch_update:
                    scheduler.step()
            if ((step + 1) % cfg.default.verbose_step == 0) or ((step + 1) == len(train_loader)):
                description = f'epoch {epoch} loss: {losses.avg:.4f}'
                pbar.set_description(description)

    if scheduler is not None and not schd_batch_update:
        scheduler.step()

    return losses.avg


def valid_fn(cfg, epoch, model, loss_fn, val_loader, device, writer, logger, scheduler=None, schd_loss_update=False):
    model.eval()
    losses = AverageMeter()
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        image_preds = model(imgs)  # output = model(input)
        # print(image_preds.shape, exam_pred.shape)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]

        loss = loss_fn(image_preds, image_labels)
        losses.update(loss.item(), cfg.default.valid_bs)

        if ((step + 1) % cfg.default.verbose_step == 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {losses.avg:.4f}'
            pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    # logger.info('validation multi-class accuracy = {:.4f}'.format(
    #     (image_preds_all == image_targets_all).mean()))

    # writer.add_scalar('valid_loss', losses.avg, epoch)
    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(losses.avg)
        else:
            scheduler.step()
    return losses.avg, image_preds_all, image_targets_all


@hydra.main(config_name='config')
def main(cfg):
    logger = init_logger(os.path.join(os.getcwd(), 'train.log'))
    writer = SummaryWriter(log_dir='./logs')

    train = pd.read_csv(cfg.common.train_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)
    seed_everything(cfg.default.seed)

    shutil.copyfile('../../../transform.py', os.path.join(os.getcwd(), 'transform.py'))

    folds = StratifiedKFold(
        n_splits=cfg.default.fold_num, shuffle=True, random_state=cfg.default.seed).split(np.arange(train.shape[0]), train.label.values)

    oof_df = pd.DataFrame()

    for fold, (trn_idx, val_idx) in enumerate(folds):
        logger.info(f"========== fold: {fold} training ==========")

        print(len(trn_idx), len(val_idx))
        train_loader, val_loader = prepare_dataloader(
            cfg, train, trn_idx, val_idx, data_root=cfg.common.img_path)

        logger.info(cfg.default.model_arch)
        model = select_model(cfg.default.model_arch, train.label.nunique()).to(device)
        scaler = GradScaler()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.shd_para.lr, weight_decay=cfg.default.weight_decay)

        scheduler = get_scheduler(cfg, optimizer)
        # ---------------
        # SAMを使いたい
        # ---------------
        # base_optimizer = torch.optim.SGD
        # optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, lr=0.1, momentum=0.9, weight_decay=0.0005)
        # scheduler = None

        loss_tr = nn.CrossEntropyLoss().to(device)  # MyCrossEntropyLoss().to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)

        best_score = 0.
        for epoch in range(cfg.default.epochs):
            train_loss = train_fn(
                cfg, epoch, model, loss_tr, optimizer, train_loader, scaler, device, writer, scheduler=scheduler, schd_batch_update=False)
            with torch.no_grad():
                valid_loss, valid_preds, valid_labels = valid_fn(
                    cfg, epoch, model, loss_fn, val_loader, device, writer, logger, scheduler=None, schd_loss_update=False)
            score = get_score(valid_labels, valid_preds)
            logger.info(f'Epoch {epoch+1} - avg_train_loss: {train_loss:.4f}  avg_val_loss: {valid_loss:.4f}')
            logger.info(f'Epoch {epoch+1} - Accuracy: {score}')
            if score > best_score:
                best_score = score
                logger.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
                torch.save({'model': model.state_dict(), 'preds': valid_preds}, f'{cfg.default.model_arch}_fold{fold}_best.pth')
            torch.save(model.state_dict(), f'{cfg.default.model_arch}_fold_{fold}_{epoch}')

        check_point = torch.load(f'{cfg.default.model_arch}_fold{fold}_best.pth')
        _oof_df = check_point['preds']
        oof_df = pd.concat([oof_df, _oof_df])
        logger.info(f"========== fold: {fold} result ==========")
        score = get_score(valid_labels, _oof_df.values)
        logger.info(f'Score - Accuracy: {score}')

        del model, optimizer, train_loader, val_loader, scaler, scheduler
        torch.cuda.empty_cache()
    
    logger.info("========== CV ==========")
    score = get_score(train.label, oof_df.values)
    oof_df.to_csv('oof_df.csv', index=False)
    logger.info(f'Score: {score:<.5f}')
    writer.close()


if __name__ == '__main__':
    main()
