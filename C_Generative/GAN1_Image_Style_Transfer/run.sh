#!/bin/bash


# 定义任务、模型和训练样本数量的不同配置
TASKS_NAME_LIST=("apple2orange" "monet2photo" "cezanne2photo" "ukiyoe2photo" "vangogh2photo" "ae_photos")

# gpu选择
GPU_IDS=${gpu_ids:-"0"}
# 训练任务名称
TASK_NAME=${task_name:-"apple2orange"}
# 学习率
LR=${lr:-0.0002}
# batch size
BATCH_SIZE=${batch_size:-1}
# 训练轮数
EPOCHS=${epochs:-300}
# 图片尺寸
SIZE=${size:-256}
# 多少个epoch后开始衰减
DECAY_EPOCH=${decay_epoch:-100}


python trainrun.py \
--gpu_ids $GPU_IDS \
--task_name $TASK_NAME \
--lr $LR \
--batch_size $BATCH_SIZE \
--epochs $EPOCHS \
--size $SIZE \
--decay_epoch $DECAY_EPOCH

# 运行脚本
# GPU_IDS="0" TASK_NAME="apple2orange" LR=0.0002 BATCH_SIZE=1 EPOCHS=300 SIZE=256 DECAY_EPOCH=100 bash run.sh