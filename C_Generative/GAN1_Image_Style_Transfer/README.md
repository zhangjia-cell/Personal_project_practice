# 代码说明

## **1. 项目简介**

本项目是一个关于图像风格迁移的练习项目

* **项目名称**：图像风格迁移

* **项目作者**：张嘉

* **项目时间**：2025年8月28日



## 2. CycleGAN 介绍





## 3. 环境配置

```python
# 安装Conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
~/miniconda3/bin/conda init
source ~/.bashrc
conda --version

# 创建conda环境
conda create --name GAN1_IST_test python==3.10
conda activate GAN1_IST_test
# 创建torch 和 torchvision
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# 导入 tqdm
conda install -c conda-forge tqdm
# 导入 transformers
conda install -c conda-forge transformers
# 导入 scipy库
conda install -c conda-forge scipy
# 重装 numpy 
conda install -c conda-forge numpy scikit-learn --force-reinstall
conda install -c conda-forge numpy scikit-learn transformers --force-reinstall

```



## 4. 调参设置

### 4.1 代码传递

```python
# 从本地传递到服务器

scp -P 12400 "F:\C_Code_AI_Program\C_Generative\GAN1_Image_Style_Transfer.zip" zhangjia@222.27.87.23:/home/zhangjia/Personal_Programs/GAN1_Image_Style_Transfer
# 从本地git到Github，在从Github上clone到服务器


```



### 4.2. 运行脚本

```python
# 运行代码(Windows本地测试)
python -u "f:\C_Code_AI_Program\C_Generative\GAN1_Image_Style_Transfer\trainrun.py" --gpu_ids 0 --task_name apple2orange --lr 0.0002 --batch_size 1 --epochs 1000 --size 256 --decay_epoch 60
# 运行代码(Linux服务运行)
GPU_IDS=0 TASK_NAME=apple2orange LR=0.0002 BATCH_SIZE=1 EPOCHS=1000 SIZE=256 DECAY_EPOCH=60 bash run.sh
```

### 4.3 参数设置

```python
# gpu卡号
gpu_ids:-"0"
# 训练任务名称
task_name:-"apple2orange"
# 学习率
lr:-0.0002
# batch size
batch_size:-1
# 训练轮数
epochs:-300
# 图片尺寸
size:-256
# 多少个epoch后开始衰减
decay_epoch:-60
```



