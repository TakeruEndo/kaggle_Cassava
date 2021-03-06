import sys
import os
import numpy as np

import cv2
import torch
from torch.utils.data import Dataset

sys.path.append('../../../FMix_master')
sys.path.append('FMix_master')
from fmix import sample_mask, make_low_freq_image, binarise_mask
from transforms import transform


def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    return im_rgb


def cut_color(image, min_, max_):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_min = np.array(min_, np.uint8)
    color_max = np.array(max_, np.uint8)
    mask = cv2.inRange(image_hsv, color_min, color_max)
    mask = cv2.bitwise_not(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img


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
    def __init__(self, cfg, type_, df, data_root,
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
        self.type_ = type_
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.except_transform = transform.get_except_transforms(cfg)
        self.data_root = data_root
        self.do_fmix = do_fmix
        self.do_mixup = do_mixup
        self.fmix_params = {
            'alpha': 1.,
            'decay_power': 3.,
            'shape': (self.cfg.default.img_size, self.cfg.default.img_size),
            'max_soft': True,
            'reformulate': False
        }
        self.do_cutmix = do_cutmix
        self.do_mixup = do_mixup
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
            try:
                img = self.transforms(image=img)['image']
            except:
                img = self.except_transform(image=img)['image']
        
        random_num = np.random.uniform(0., 1., size=1)[0]

        if self.cfg.da.do_fmix and random_num > 0.25 and random_num < 0.50 and self.type_ == 'train':
            with torch.no_grad():
                # lam, mask = sample_mask(**self.fmix_params)
                lam = np.clip(np.random.beta(self.fmix_params['alpha'], self.fmix_params['alpha']), 0.6, 0.7)

                # Make mask, get mean / std
                mask = make_low_freq_image(
                    self.fmix_params['decay_power'], self.fmix_params['shape'])
                mask = binarise_mask(
                    mask, lam, self.fmix_params['shape'], self.fmix_params['max_soft'])

                fmix_ix = np.random.choice(self.df.index, size=1)[0]
                fmix_img = get_img(
                    "{}/{}".format(self.data_root, self.df.iloc[fmix_ix]['image_id']))

                if self.transforms:
                    try:
                        fmix_img = self.transforms(image=fmix_img)['image']
                    except:
                        fmix_img = self.except_transform(image=fmix_img)['image']                    

                mask_torch = torch.from_numpy(mask)

                # mix image
                img = mask_torch * img + (1. - mask_torch) * fmix_img

                # assert self.output_label==True and self.one_hot_label==True

                # mix target
                rate = mask.sum() / self.cfg.default.img_size / self.cfg.default.img_size
                target = rate * target + (1. - rate) * self.labels[fmix_ix]
                # print(target, mask, img)
                # assert False
        if self.cfg.da.do_cutmix and random_num > 0.75 and random_num < 1.0 and self.type_ == 'train':
            # print(img.sum(), img.shape)
            with torch.no_grad():
                cmix_ix = np.random.choice(self.df.index, size=1)[0]
                cmix_img = get_img(
                    "{}/{}".format(self.data_root, self.df.iloc[cmix_ix]['image_id']))
                if self.transforms:
                    try:
                        cmix_img = self.transforms(image=cmix_img)['image']
                    except:
                        cmix_img = self.except_transform(image=cmix_img)['image']                      

                lam = np.clip(np.random.beta(
                    self.cutmix_params['alpha'], self.cutmix_params['alpha']), 0.3, 0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox(
                    (self.cfg.default.img_size, self.cfg.default.img_size), lam)

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]

                rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                            (self.cfg.default.img_size * self.cfg.default.img_size))
                target = rate * target + (1. - rate) * self.labels[cmix_ix]
        if self.cfg.da.do_mixup and random_num > 0.50 and random_num < 0.75 and self.type_ == 'train':
            """
            Reference: https://github.com/karaage0703/pytorch-example/blob/master/pytorch_data_preprocessing.ipynb
            """
            with torch.no_grad():
                mix_ix = np.random.choice(self.df.index, size=1)[0]
                mix_img = get_img(
                    "{}/{}".format(self.data_root, self.df.iloc[mix_ix]['image_id']))
                if self.transforms:
                    try:
                        mix_img = self.transforms(image=mix_img)['image']
                    except:
                        mix_img = self.except_transform(image=mix_img)['image']                      
                    

                lam = np.random.beta(1.0, 1.0)
                img = lam * img + (1 - lam) * mix_img

                target = lam * target + (1. - lam) * self.labels[mix_ix]

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
