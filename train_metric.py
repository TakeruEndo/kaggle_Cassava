from utils import seed_everything
from transforms.transform import get_train_transforms, get_valid_transforms
from dataset import CassavaDataset
from Model import efficientnet, metric_learning
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

from pytorch_metric_learning import losses, miners, samplers, trainers, testers
from pytorch_metric_learning.utils import common_functions
import pytorch_metric_learning.utils.logging_presets as logging_presets
import logging
import umap
from cycler import cycler
import record_keeper
import pytorch_metric_learning

sys.path.append('FMix-master')


def prepare_dataset(cfg, df, trn_idx, val_idx, data_root='../input/cassava-leaf-disease-classification/train_images/'):

    train_ = df.loc[trn_idx, :].reset_index(drop=True)
    valid_ = df.loc[val_idx, :].reset_index(drop=True)

    train_ds = CassavaDataset(
        cfg, train_, data_root, transforms=get_train_transforms(cfg), output_label=True, one_hot_label=False, do_fmix=False, do_cutmix=False)
    valid_ds = CassavaDataset(
        cfg, valid_, data_root, transforms=get_valid_transforms(cfg), output_label=True)

    return train_ds, valid_ds


def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
    logging.info("UMAP plot for the {} split and label set {}".format(
        split_name, keyname))
    label_set = np.unique(labels)
    num_classes = len(label_set)
    fig = plt.figure(figsize=(20, 15))
    plt.gca().set_prop_cycle(cycler(
        "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]))
    for i in range(num_classes):
        idx = labels == label_set[i]
        plt.scatter(umap_embeddings[idx, 0], umap_embeddings[idx,
                                                             1], s=20, marker="o", alpha=0.5, label=str(i))
    plt.show()


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
        train_dataset, val_dataset = prepare_dataset(
            cfg, train, trn_idx, val_idx, data_root=cfg.common.img_path)

        # Set trunk model and replace the softmax layer with an identity function
        trunk = metric_learning.CassvaImgTrunk(
            cfg.default.model_arch, train.label.nunique(), pretrained=True).to(device)
        trunk = torch.nn.DataParallel(trunk.to(device))

        embedder = torch.nn.DataParallel(
            metric_learning.CassvaImgEmbedder(256))
        classifier = torch.nn.DataParallel(
            metric_learning.CassvaImgClassifier(64, 5, 200)).to(device)

        trunk_optimizer = torch.optim.Adam(
            trunk.parameters(), lr=cfg.default.lr, weight_decay=0.0001)
        embedder_optimizer = torch.optim.Adam(
            embedder.parameters(), lr=cfg.default.lr, weight_decay=0.0001)
        classifier_optimizer = torch.optim.Adam(
            classifier.parameters(), lr=cfg.default.lr, weight_decay=0.0001)

        # Set the loss function
        loss = losses.TripletMarginLoss(margin=0.1)

        # Set the classification loss:
        classification_loss = torch.nn.CrossEntropyLoss()

        # Set the mining function
        miner = miners.MultiSimilarityMiner(epsilon=0.1)

        # Set the dataloader sampler
        sampler = samplers.MPerClassSampler(
            train_dataset.labels, m=4, length_before_new_iter=len(train_dataset))

        # Package the above stuff into dictionaries.
        models = {"trunk": trunk, "embedder": embedder,
                  "classifier": classifier}
        optimizers = {"trunk_optimizer": trunk_optimizer,
                      "embedder_optimizer": embedder_optimizer, "classifier_optimizer": classifier_optimizer}
        loss_funcs = {"metric_loss": loss,
                      "classifier_loss": classification_loss}
        mining_funcs = {"tuple_miner": miner}

        # We can specify loss weights if we want to. This is optional
        loss_weights = {"metric_loss": 0.5, "classifier_loss": 1}

        record_keeper, _, _ = logging_presets.get_record_keeper(
            "example_logs", "example_tensorboard")
        hooks = logging_presets.get_hook_container(record_keeper)
        dataset_dict = {"val": val_dataset}
        model_folder = "output"

        # Create the tester
        tester = testers.GlobalEmbeddingSpaceTester(end_of_testing_hook=hooks.end_of_testing_hook,
                                                    visualizer=umap.UMAP(),
                                                    visualizer_hook=visualizer_hook,
                                                    dataloader_num_workers=2)

        end_of_epoch_hook = hooks.end_of_epoch_hook(tester,
                                                    dataset_dict,
                                                    model_folder,
                                                    test_interval=1)

        trainer = trainers.TrainWithClassifier(models,
                                               optimizers,
                                               cfg.default.train_bs,
                                               loss_funcs,
                                               mining_funcs,
                                               train_dataset,
                                               sampler=sampler,
                                               dataloader_num_workers=2,
                                               loss_weights=loss_weights,
                                               end_of_iteration_hook=hooks.end_of_iteration_hook,
                                               end_of_epoch_hook=end_of_epoch_hook)

        trainer.train(num_epochs=cfg.default.num_epochs)


if __name__ == '__main__':
    main()
