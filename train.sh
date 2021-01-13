python train.py \
default.model_arch='tf_efficientnet_b6_ns' \
default.train_bs=4 \
default.valid_bs=4 \
default.img_size=528

python train.py \
default.model_arch='tf_mixnet_s' \
default.img_size=528

python train.py \
default.model_arch='tf_efficientnet_b4_ns' \
default.img_size=528

python train.py
default:
  model_arch: 'vit_base_patch16_384'