<a href="https://www.kaggle.com/c/cassava-leaf-disease-classification"><img src=https://github.com/TakeruEndo/kaggle_Cassava/blob/main/imgs/img.png></a>

## 実験結果
| model_name            | image_size | loss                 | optimizer | schduler | data_aug | best_score_loss | best_score |
| --------------------- | ---------- | -------------------- | --------- | -------- | -------- | --------------- | ---------- |
| tf_efficientnet_b4_ns | 512        | FocalCosineLoss      | adam      | ConsAnne | 1        | 0.1355          | 0.89276    |
| tf_efficientnet_b4_ns | 512        | CrossEntropy         | adam      | ConsAnne | 1        | 0.3251          | 0.89462    |
| tf_efficientnet_b4_ns | 600        | CrossEntropy         | adam      | ConsAnne | 1        | 0.3103          | 0.89626    |
| tf_efficientnet_b4_ns | 512        | CrossEntropy         | adam      | ConsAnne | 1        | 0.3251          | 0.89462    |
| tf_efficientnet_b4_ns | 512        | CrossEntropy         | adam      | ConsAnne | 2        | 0.3178          | 0.89393    |
| tf_efficientnet_b4_ns | 600        | LabelSmmothingLoss   | adam      | ConsAnne | 1        | 0.3100          | 0.89860    |
| tf_efficientnet_b5_ns | 512        | CrossEntropy         | adam      | ConsAnne | 1        | 0.3103          | 0.89700    |
| tf_efficientnet_b5_ns | 600        | CrossEntropy         | adam      | ConsAnne | 1        | 0.3103          | 0.89603    |
| tf_efficientnet_b5_ns | 512        | LabelSmmothingLoss   | adam      | ConsAnne | 1        | 0.3370          | 0.89860    |
| tf_efficientnet_b5_ns | 512        | TaylorCrossEntropy   | adam      | ConsAnne | 1        | 0.3095          | 0.89766    |
| tf_efficientnet_b5_ns | 512        | SymmtricCrossENtropy | adam      | ConsAnne | 1        | 0.3574          | 0.89580    |
| tf_efficientnet_b6    | 528        | CrossEntropy         | adam      | ConsAnne | 1        | 0.3407          | 0.89042    |
| tf_efficientnet_b6_ns | 528        | CrossEntropy         | adam      | ConsAnne | 1        | 0.3294          | 0.88902    |
| tf_mixnet_s           | 512        | CrossEntropy         | adam      | ConsAnne | 1        | 0.3383          | 0.88505    |
| vit_base_patch16_38   | 384        | CrossEntropy         | adam      | ConsAnne | 1        | 0.7168          | 0.73808    |

## TODO:
- [ ] Pseud label learning
- [ ] [Meta Pseud Label](https://github.com/TakeruEndo/paper-survey/issues/65)
- [ ] Focal_cosine loss
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

