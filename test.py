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
# from efficientnet_pytorch import EfficientNet
from scipy.ndimage.interpolation import zoom
from catalyst.data.sampler import BalanceClassSample
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append('FMix-master')
from Model import efficientnet
from dataset import CassavaDataset
from transforms.transform import get_train_transforms, get_valid_transforms, get_inference_transforms
from utils import seed_everything


def inference_one_epoch(model, data_loader, device):
    model.eval()

    image_preds_all = []

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()

        image_preds = model(imgs)  # output = model(input)
        image_preds_all += [torch.softmax(image_preds,
                                          1).detach().cpu().numpy()]

    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all


@hydra.main(config_name='config')
def main(cfg):
    train = pd.read_csv(cfg.common.train_path)

    seed_everything(cfg.default.seed)

    folds = StratifiedKFold(n_splits=cfg.default.fold_num).split(
        np.arange(train.shape[0]), train.label.values)

    for fold, (trn_idx, val_idx) in enumerate(folds):
        # we'll train fold 0 first
        if fold > 0:
            break

        print('Inference fold {} started'.format(fold))

        valid_ = train.loc[val_idx, :].reset_index(drop=True)
        valid_ds = CassavaDataset(valid_, '../input/cassava-leaf-disease-classification/train_images/',
                                  transforms=get_inference_transforms(), output_label=False)

        test = pd.DataFrame()
        test['image_id'] = list(os.listdir(
            '../input/cassava-leaf-disease-classification/test_images/'))
        test_ds = CassavaDataset(test, '../input/cassava-leaf-disease-classification/test_images/',
                                 transforms=get_inference_transforms(), output_label=False)

        val_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=cfg.inf.valid_bs,
            num_workers=cfg.inf.num_workers,
            shuffle=False,
            pin_memory=False,
        )

        tst_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=cfg.inf.valid_bs,
            num_workers=cfg.inf.num_workers,
            shuffle=False,
            pin_memory=False,
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = efficientnet.CustomEfficientNet(
            cfg.default.model_arch, train.label.nunique()).to(device)

        val_preds = []
        tst_preds = []

        # for epoch in range(CFG['epochs']-3):
        for i, epoch in enumerate(cfg.ing.used_epochs):
            model.load_state_dict(torch.load(
                '../input/cassava-efficientnet-training/{}_fold_{}_{}'.format(cfg.default.model_arch, fold, epoch)))

            with torch.no_grad():
                for _ in range(cfg.inf.tta):
                    val_preds += [cfg.inf.weights[i] / sum(
                        cfg.inf.weights) / cfg.inf.tta * inference_one_epoch(model, val_loader, device)]
                    tst_preds += [cfg.inf.weights[i] / sum(
                        cfg.inf.weights) / cfg.inf.tta * inference_one_epoch(model, tst_loader, device)]

        val_preds = np.mean(val_preds, axis=0)
        tst_preds = np.mean(tst_preds, axis=0)

        print('fold {} validation loss = {:.5f}'.format(
            fold, log_loss(valid_.label.values, val_preds)))
        print('fold {} validation accuracy = {:.5f}'.format(
            fold, (valid_.label.values == np.argmax(val_preds, axis=1)).mean()))

        del model
        torch.cuda.empty_cache()
