from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop, RandomSunFlare,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize,
    ISONoise, RandomResizedCrop, RandomShadow
)

from transforms.grid_mask import GridMask


def get_train_transforms(cfg):
    da_version = cfg.default.da_version
    if da_version == 1:
        return Compose([
            RandomResizedCrop(cfg.default.img_size, cfg.default.img_size),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            RandomSunFlare(p=0.5, src_color=[160, 160, 160]),
            OneOf([
                ISONoise(p=1.0),
                GaussNoise(p=0.8),
                IAAAdditiveGaussianNoise(p=0.8),
                CLAHE(p=0.7)
            ], p=1.0),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            OneOf([
                Cutout(num_holes=10, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.8),
                CoarseDropout(p=0.8),
            ], p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)
    elif da_version == 2:
        return Compose([
            RandomResizedCrop(cfg.default.img_size, cfg.default.img_size),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            OneOf([
                RandomShadow(p=0.8),
                RandomSunFlare(src_color=[160, 160, 160], p=0.8),
            ], p=1.0),
            ISONoise(p=0.5),
            OneOf([
                GaussNoise(p=0.8),
                IAAAdditiveGaussianNoise(p=0.8),
                MotionBlur(p=1.0),
                MedianBlur(p=1.0)
            ], p=1.0),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            OneOf([
                Cutout(num_holes=10, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.8),
                CoarseDropout(p=0.8)
            ], p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.),
    elif da_version == 3:
        return Compose([
            OneOf([            
                RandomResizedCrop(cfg.default.img_size, cfg.default.img_size, p=1.),
                CenterCrop(cfg.default.img_size, cfg.default.img_size, p=1.),
            ], p=1.0),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            RandomSunFlare(src_color=[160, 160, 160], p=0.5),
            ISONoise(p=0.5),
            OneOf([
                GaussNoise(p=0.8),
                IAAAdditiveGaussianNoise(p=0.8),
                MotionBlur(p=1.0),
                MedianBlur(p=1.0)
            ], p=1.0),
            CLAHE(p=0.6), 
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            OneOf([
                Cutout(num_holes=10, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.8),
                CoarseDropout(p=0.8),
            ], p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.) 


def get_valid_transforms(cfg):
    return Compose([
        CenterCrop(cfg.default.img_size, cfg.default.img_size, p=1.),
        Resize(cfg.default.img_size, cfg.default.img_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[
                  0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


def get_inference_transforms(cfg):
    return Compose([
        RandomResizedCrop(cfg.default.img_size, cfg.default.img_size),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        HueSaturationValue(hue_shift_limit=0.2,
                           sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[
                  0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)
