<a href="https://www.kaggle.com/c/cassava-leaf-disease-classification"><img src=https://github.com/TakeruEndo/kaggle_Cassava/blob/main/imgs/img.png></a>

## 結果
🥉227/3900

## 実験結果
| model_name            | Dataset | mix_da | image_size | loss                 | optimizer | schduler | data_aug | best_score_loss | best_score | fold0 score |
| --------------------- | ---------- | -------------------- | --------- | -------- | -------- | --------------- | ---------- | ---- |---------- | -------------------- |
| RepVGG-B1g4   | 2019 + 2020 | fmix | 512       | CrossEntropy         | adam      | ConsAnne | 1        | 0.3563    |0.890660  (13epoch)     |   0.890 |
| RepVGG-B1g4 | 2020        |  mixup | 512       | CrossEntropy         | adam      | ConsAnne | 1        | 0.3519          | 0.8887    |  ---- |
| resnext50_32x4d | 2019 + 2020 | fmix | 512    | CrossEntropy | adam | ConsAnne | 1        |  0.3823    |0.88743356 (13epoch)   | 0.891 |
| resnext50_32x4d | 2020 | mixup | 512    | CrossEntropy | adam | ConsAnne | 1        |  0.3547    |0.88995 (20epoch)   | ---- |
| tf_efficientnet_b4_ns | 2019 + 2020 | fmix | 512        | CrossEntropy | adam | ConsAnne | 1        |   0.3377    |0.888572513287 (6epoch)   | ---- |
| tf_efficientnet_b4_ns | 2020 | mixup | 512        | CrossEntropy | adamp | ConsAnne | 1        |   0.3346    | 0.89205607 (9epoch)   | ---- |
| tf_efficientnet_b5_ns | 2019 + 2020 | fmix | 512        | CrossEntropy | adam | ConsAnne | 1        |   0.3423    |0.890280941 (8epoch)   | 0.8999 |
| tf_efficientnet_b5_ns | 2020 | mixup | 512        | CrossEntropy | adamp | ConsAnne | 1        |  0.3318   |0.8936915 (6epoch)   | ---  |
| tf_efficientnet_b5_ns | 2020 | mixup | 512        | CrossEntropy | adamp | ConsAnne | 1        |   0.3318    |0.89252 (5epoch)   | ---- |
| tf_mixnet_s           | 2020 | --- |512        | CrossEntropy         | adam      | ConsAnne | 1        | 0.3383          | 0.88505    | ---  |
| vit_base_patch16_38   | 2020 | --- |384        | CrossEntropy         | adam      | ConsAnne | 1        | 0.7168          | 0.73808    | ---  |
| deit_base_patch_16_224   | 2020 | --- |224      | CrossEntropy         | adam      | ConsAnne | 1        | 0.9590          | 0.6596    | ---  |


## TODO:
- [x] Pseud label learning
- [x] [Meta Pseud Label](https://github.com/TakeruEndo/paper-survey/issues/65)
- [x] Focal_cosine loss
  - [discussion](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203271)


### trainで参考にしているnotebook
1. [[CNN or Transformer]-Pytorch XLA(TPU) for Cassava](https://www.kaggle.com/piantic/cnn-or-transformer-pytorch-xla-tpu-for-cassava)
2. [Pytorch Efficientnet Baseline [Train] AMP+Aug](https://www.kaggle.com/khyeh0719/pytorch-efficientnet-baseline-train-amp-aug)
3. [Cassava / resnext50_32x4d starter [training]](https://www.kaggle.com/yasufuminakama/cassava-resnext50-32x4d-starter-training)

## Albumentations
https://albumentations.ai/docs/api_reference/augmentations/transforms/

## HYDRA
https://hydra.cc/docs/configure_hydra/workdir/

## FMix
https://github.com/ecs-vlc/FMix

## timm
https://rwightman.github.io/pytorch-image-models/models/

