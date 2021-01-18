import sys
import os
import numpy as np

import cv2
import torch
from torch.utils.data import Dataset

sys.path.append('../../../FMix_master')
sys.path.append('FMix_master')
from fmix import sample_mask, make_low_freq_image, binarise_mask


def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    # print(im_rgb)
    return im_rgb


def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


class CassavaDataset(Dataset):
    def __init__(self, cfg, df, data_root,
                 transforms=None,
                 output_label=True,
                 one_hot_label=False,
                 do_fmix=False,
                 do_cutmix=False,
                 do_mixup=False,
                 cutmix_params={
                     'alpha': 1,
                 }
                 ):

        super().__init__()
        self.cfg = cfg
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.do_fmix = do_fmix
        self.fmix_params = {
            'alpha': 1.,
            'decay_power': 3.,
            'shape': (self.cfg.default.img_size, self.cfg.default.img_size),
            'max_soft': True,
            'reformulate': False
        },
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params

        self.output_label = output_label
        self.one_hot_label = one_hot_label

        if output_label:
            self.labels = self.df['label'].values
            # print(self.labels)

            if one_hot_label is True:
                self.labels = np.eye(self.df['label'].max() + 1)[self.labels]
                # print(self.labels)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):

        # get labels
        if self.output_label:
            target = self.labels[index]

        img = get_img("{}/{}".format(self.data_root,
                                     self.df.loc[index]['image_id']))

        if self.transforms:
            img = self.transforms(image=img)['image']

        if self.do_fmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            with torch.no_grad():
                # lam, mask = sample_mask(**self.fmix_params)
                lam = np.clip(np.random.beta(
                    self.fmix_params['alpha'], self.fmix_params['alpha']), 0.6, 0.7)

                # Make mask, get mean / std
                mask = make_low_freq_image(
                    self.fmix_params['decay_power'], self.fmix_params['shape'])
                mask = binarise_mask(
                    mask, lam, self.fmix_params['shape'], self.fmix_params['max_soft'])

                fmix_ix = np.random.choice(self.df.index, size=1)[0]
                fmix_img = get_img(
                    "{}/{}".format(self.data_root, self.df.iloc[fmix_ix]['image_id']))

                if self.transforms:
                    fmix_img = self.transforms(image=fmix_img)['image']

                mask_torch = torch.from_numpy(mask)

                # mix image
                img = mask_torch * img + (1. - mask_torch) * fmix_img

                # print(mask.shape)

                # assert self.output_label==True and self.one_hot_label==True

                # mix target
                rate = mask.sum() / self.cfg.default.img_size / self.cfg.default.img_size
                target = rate * target + (1. - rate) * self.labels[fmix_ix]
                # print(target, mask, img)
                # assert False

        if self.do_cutmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            # print(img.sum(), img.shape)
            with torch.no_grad():
                cmix_ix = np.random.choice(self.df.index, size=1)[0]
                cmix_img = get_img(
                    "{}/{}".format(self.data_root, self.df.iloc[cmix_ix]['image_id']))
                if self.transforms:
                    cmix_img = self.transforms(image=cmix_img)['image']

                lam = np.clip(np.random.beta(
                    self.cutmix_params['alpha'], self.cutmix_params['alpha']), 0.3, 0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox(
                    (self.cfg.default.img_size, self.cfg.default.img_size), lam)

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]

                rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                            (self.cfg.default.img_size * self.cfg.default.img_size))
                target = rate * target + (1. - rate) * self.labels[cmix_ix]

        if self.do_mixup and np.random.uniform(0., 1., size=1)[0] > 0.5:
            with torch.no_grad():
                mix_ix = np.random.choice(self.df.index, size=1)[0]
                mix_img = get_img(
                    "{}/{}".format(self.data_root, self.df.iloc[mix_ix]['image_id']))
                if self.transforms:
                    mix_img = self.transforms(image=mix_img)['image']

                lam = np.random.beta(1.0, 1.0)
                img = lam * img + (1 - lam) * mix_img

                target = lam * target + (1. - lam) * self.labels[mix_ix]

            # print('-', img.sum())
            # print(target)
            # assert False

        # do label smoothing
        # print(type(img), type(target))
        if self.output_label:
            return img, target
        else:
            return img


class CassavaTestDataset(Dataset):
    def __init__(
        self, df, data_root, transforms=None, output_label=True
    ):

        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):

        # get labels
        if self.output_label:
            target = self.df.iloc[index]['label']

        path = "{}/{}".format(self.data_root, self.df.iloc[index]['image_id'])

        img = get_img(path)

        if self.transforms:
            img = self.transforms(image=img)['image']

        # do label smoothing
        if self.output_label:
            return img, target
        else:
            return img
