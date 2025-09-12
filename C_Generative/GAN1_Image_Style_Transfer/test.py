import os
import torchvision.transforms as transforms
import argparse
from transformers import Trainer, HfArgumentParser, TrainingArguments
from dataclasses import dataclass
import zipfile
import requests
from tqdm import tqdm
import time 
import logging
from datetime import datetime
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch 
from models import Generator, Discriminator
from utils import weights_init_normal, ReplayBuffer, LambdaLR, tensor2image
from datasets import ImageDataset
import itertools
from trainrun import parse_args

# --------------------------------------------------------------------------------------------------------------------- #
# 1.参数设置------*[需要修改]*
# --------------------------------------------------------------------------------------------------------------------- #
args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG_A2B = Generator().to(device)  # 生成器 A2B（将A风格图像转换为B风格图像）
netG_B2A = Generator().to(device)  # 生成器 B2A（将B风格图像转换为A风格图像）

netG_A2B.load_state_dict(torch.load("C_Generative/GAN1_Image_Style_Transfer/saved_models/netG_A2B.pth", map_location=device))
netG_B2A.load_state_dict(torch.load("C_Generative/GAN1_Image_Style_Transfer/saved_models/netG_B2A.pth", map_location=device))

netG_A2B.eval()
netG_B2A.eval()


# 初始化输入张量和标签（占位符，后续会被真实数据覆盖）
input_A = torch.ones([1, 3, args.size, args.size], dtype=torch.float).to(device)  # 输入A
input_B = torch.ones([1, 3, args.size, args.size], dtype=torch.float).to(device)  # 输入B


# 数据加载器
transforms_ = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

