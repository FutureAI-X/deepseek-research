# 如果你要限制计算卡编号，请在这里设置，例如只使用 cuda:1-3，如果不用限制，就删除下面这行
# export CUDA_VISIBLE_DEVICES=1,2,3

accelerate launch \
    --config_file accelerate_config.yaml \
    train_r1.py \
    --config trl_config.yaml