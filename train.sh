# python train.py \
# default.model_arch='tf_efficientnet_b6_ns' \
# default.train_bs=4 \
# default.valid_bs=4 \
# default.img_size=528

# python train.py \
# default.model_arch='tf_mixnet_s' \
# default.train_bs=16 \
# default.valid_bs=16

# python train.py \
# default.model_arch='tf_efficientnet_b4_ns' \
# default.loss_fn='FocalCosineLoss'

# python train.py \
# default.model_arch='vit_base_patch16_384' \
# default.img_size=384

# memoryが足りない
# python train.py \
# default.model_arch='tf_efficientnet_l2_ns_475' \
# default.train_bs=2 \
# default.valid_bs=2 \
# default.img_size=475

# python train.py \
# default.model_arch='tf_efficientnet_b6' \
# default.train_bs=4 \
# default.valid_bs=4 \
# default.img_size=528

# python train.py \
# default.model_arch='deit_base_patch16_224' \
# default.img_size=224

# python train.py \
# default.model_arch='tf_efficientnet_b5_ns' \
# default.train_bs=4 \
# default.valid_bs=4 \
# default.img_size=600

# python train.py \
# default.model_arch='resnext50_32x4d' \
# default.img_size=600

# python train.py \
# default.model_arch='tf_efficientnet_b4_ns' \
# default.da_version=2

# python train.py \
# default.model_arch='tf_efficientnet_b5_ns' \

# python train.py \
# default.model_arch='tf_efficientnet_b5_ns' \
# default.loss_fn='LabelSmoothingLoss'

# python train.py \
# default.model_arch='vit_base_patch16_384' \
# default.img_size=384

# python train.py \
# default.model_arch='tf_efficientnet_b5_ns' \
# default.loss_fn='TaylorCrossEntropyLoss'

# python train.py \
# default.model_arch='tf_efficientnet_b5_ns' \
# default.loss_fn='SymmetricCrossEntropy'

# python train.py \
# default.model_arch='tf_efficientnet_b4_ns' \
# default.loss_fn='LabelSmoothingLoss' \
# default.img_size=600

# python train.py \
# default.model_arch='tf_efficientnet_b3_ns' \

# python train.py \
# default.model_arch='tf_efficientnet_b3_ns' \
# default.img_size=600
python train.py \
default.model_arch='resnext50_32x4d' \
default.loss_fn='BiTemperedLogisticLoss' \
default.img_size=512 \

python train.py \
default.model_arch='resnext50_32x4d' \
default.loss_fn='LabelSmoothingLoss' \
default.img_size=512 \

python train.py \
default.model_arch='tf_efficientnet_b4_ns' \
default.img_size=600

python train.py \
default.model_arch='tf_efficientnet_b4_ns' \
default.loss_fn='BiTemperedLogisticLoss' \
default.img_size=600

# python train.py \
# default.model_arch='tf_efficientnet_b4_ns' \
# default.loss_fn='LabelSmoothingLoss' \
# default.img_size=600 \
# shd_para.scheduler='CosineAnnealingLR'

# python train.py \
# default.model_arch='tf_efficientnet_b4_ns' \
# default.loss_fn='LabelSmoothingLoss' \
# default.img_size=600 \
# default.da_version=3