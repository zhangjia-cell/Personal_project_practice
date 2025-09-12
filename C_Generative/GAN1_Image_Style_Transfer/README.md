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

### 4.1. 运行脚本

```python
# 训练脚本
python trainrun.py --task_name apple2orange --lr 0.0002 --batch_size 1 --epochs 1000 --size 256 --decay_epoch 60
```

### 4.2 参数设置

```python
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



