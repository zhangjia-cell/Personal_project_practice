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


# --------------------------------------------------------------------------------------------------------------------- #
# 1.参数设置------*[可以增补]*
# --------------------------------------------------------------------------------------------------------------------- #
# 可调参数
@dataclass
class OurArguments:
    # 显卡设置
    gpu_ids: str = "0"
    # 任务名称
    task_name : str = "apple2orange"
    # 学习率
    lr: float = 0.0002
    # batch size
    batch_size: int = 1
    # 训练轮数
    epochs: int = 300
    # 图片尺寸
    size: int = 256
    # 多少个epoch后开始衰减
    decay_epoch: int = 60
    # 下载地址
    DOWNLOAD_DIR: str = "C_Generative/GAN1_Image_Style_Transfer/Data"
    LOGS_DIR: str = "C_Generative/GAN1_Image_Style_Transfer"
    MODELS_DIR: str = "C_Generative/GAN1_Image_Style_Transfer/Models"
    Linux_Dir: str = "/home/zhangjia/Personal_Programs"

# --------------------------------------------------------------------------------------------------------------------- #
def gpuchose(args):
    device = torch.device(f"cuda:{args.gpu_ids}" if torch.cuda.is_available() else "cpu") # 设备选择
    return device
# --------------------------------------------------------------------------------------------------------------------- #

device = gpuchose(OurArguments())
# 传参设置
def parse_args():
	parser = HfArgumentParser(OurArguments)
	args = parser.parse_args_into_dataclasses()[0]
	return args

# --------------------------------------------------------------------------------------------------------------------- #
# 2.数据下载------*[可以增补]*
# --------------------------------------------------------------------------------------------------------------------- #
# 设置下载函数
def download_and_extract_zip(url: str, task_name: str, save_path: str, logger: logging.Logger):
    data_folder = os.path.join(save_path, task_name)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)
        logger.info(f"已创建文件夹: {data_folder}")
    zip_path = os.path.join(os.path.dirname(data_folder), f"{task_name}.zip")

    if os.path.exists(data_folder) and os.listdir(data_folder):
        logger.info(f"数据集 {task_name} 已存在，跳过下载。")
        return

    logger.info(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"-" * 150)
    logger.info(f"开始下载数据: {task_name} ...")

    # ---- 下载带进度条 ----
    with requests.get(url, stream=True, timeout=100) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024,
            desc=f"Downloading {task_name}"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    logger.info(f"数据: {task_name} 下载完成。")
    logger.info(f"-" * 150)
    # ---- 解压带进度条 ----
    logger.info(f"开始解压数据: {task_name} ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.infolist()
            with tqdm(total=len(members), unit="file", desc=f"Extracting {task_name}") as pbar:
                for m in members:
                    zf.extract(m, path=save_path)
                    pbar.update(1)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        logger.info(f"数据: {task_name} 解压完成。")
    except zipfile.BadZipFile:
        logger.error("解压失败：无效的 ZIP 文件：%s", zip_path)
        raise
    except Exception as e:
        logger.exception("解压失败：%s", e)
        raise

def download_data(args, task_name, save_path, logger=None):
    task_url = f"https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/{task_name}.zip"

    match task_name:
        # 数据集
        case "apple2orange" | "monet2photo" | "cezanne2photo" | "ukiyoe2photo" | "vangogh2photo" | "ae_photos":
            # 下载数据集
            download_and_extract_zip(task_url, task_name, save_path=save_path, logger=logger)
        case _:
            if logger:
                logger.error(f"未知任务名称：{task_name}")

# --------------------------------------------------------------------------------------------------------------------- #
# 3.日志设置------*[需更改]*
# --------------------------------------------------------------------------------------------------------------------- #
def setup_logging(args, log_dir="Logs"):
    # 创建日志文件夹
    # -------------------------------- #
    args.LOGS_DIR = os.path.join(args.Linux_Dir, args.LOGS_DIR) # linux版本：
    # -------------------------------- #
    log_dir = os.path.join(args.LOGS_DIR, log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建日志文件名---包含时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"training_{timestamp}.log")

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'), #输出到文件 
            logging.StreamHandler() # 同时输出到控制台
        ]
    )

    return logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------------------------------- #
# 4.网络设置------*[]*
# --------------------------------------------------------------------------------------------------------------------- #
def etwork_Settings():
    # 将A风格转换为B风格
    netG_A2B = Generator().to(device)  # 生成器 A2B
    # 将B风格转换为A风格
    netG_B2A = Generator().to(device)  # 生成器 B2A
    # 判别器A（判断输入图像是否为真实的A风格图像）
    netD_A = Discriminator().to(device)  # 判别器 A
    # 判别器B（判断输入图像是否为真实的B风格图像）
    netD_B = Discriminator().to(device)  # 判别器 B
    # 初始化权重
    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)
    return netG_A2B, netG_B2A, netD_A, netD_B

# --------------------------------------------------------------------------------------------------------------------- #
# 5.损失设置------*[]*
# --------------------------------------------------------------------------------------------------------------------- #
def loss_functions():
    loss_GAN = torch.nn.MSELoss().to(device)  # GAN 损失(最小二乘损失)
    loss_cycle = torch.nn.L1Loss().to(device)  # 循环一致性损失[生成与原图像的L1距离]
    loss_identity = torch.nn.L1Loss().to(device)  # 恒等损失[生成与输入图像的L1距离]
    return loss_GAN, loss_cycle, loss_identity

# --------------------------------------------------------------------------------------------------------------------- #
# 6.优化器设置------*[]*
# --------------------------------------------------------------------------------------------------------------------- #
# 生成器优化器

def generative_optimizers(args):
    netG_A2B, netG_B2A, netD_A, netD_B = etwork_Settings()
    # 生成器优化器
    opt_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=args.lr, betas=(0.5, 0.9999))
    # 判别器A（判断输入图像是否为真实的A风格图像）
    opt_DA = torch.optim.Adam(netD_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # 判别器B（判断输入图像是否为真实的B风格图像）
    opt_DB = torch.optim.Adam(netD_B.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # 学习率调度器
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=LambdaLR(args.epochs, 0, args.decay_epoch).step)
    lr_scheduler_DA = torch.optim.lr_scheduler.LambdaLR(opt_DA, lr_lambda=LambdaLR(args.epochs, 0, args.decay_epoch).step)
    lr_scheduler_DB = torch.optim.lr_scheduler.LambdaLR(opt_DB, lr_lambda=LambdaLR(args.epochs, 0, args.decay_epoch).step)
    return opt_G, opt_DA, opt_DB, lr_scheduler_G, lr_scheduler_DA, lr_scheduler_DB

# --------------------------------------------------------------------------------------------------------------------- #
# 7.定义训练部分------*[]*
# --------------------------------------------------------------------------------------------------------------------- #
def train_model():
    # --------------------------------------------------------- #
    # 解析参数
    args = parse_args()
    # --------------------------------------------------------- #

    # 创建日志对象
    logger = setup_logging(args)
    # ------------------------------------------------------- #

    # 加载数据集
    # ------------------------------------------------------- #
    # Linux版本：
    # ------------------------------------------------------- #
    args.DOWNLOAD_DIR = os.path.join(args.Linux_Dir, "C_Generative/GAN1_Image_Style_Transfer/Data")
    args.MODELS_DIR = os.path.join(args.Linux_Dir, args.MODELS_DIR)
    # ------------------------------------------------------- #
    if not os.path.exists(args.MODELS_DIR): 
        os.makedirs(args.MODELS_DIR, exist_ok=True)
        logger.info(f"已创建文件夹: {args.MODELS_DIR}")

    # 下载数据
    download_data(args, args.task_name, args.DOWNLOAD_DIR, logger=logger)
    logger.info("-" * 150)
    logger.info(f"开始训练模型 ...")
    logger.info("-" * 150)

    # 初始化输入张量和标签（占位符，后续会被真实数据覆盖）
    input_A = torch.ones([1, 3, args.size, args.size], dtype=torch.float).to(device)  # 输入A
    input_B = torch.ones([1, 3, args.size, args.size], dtype=torch.float).to(device)  # 输入B
    # 真实标签和假标签
    label_real = torch.ones([1], dtype=torch.float, requires_grad=False).to(device=device)  # 真实标签
    label_fake = torch.zeros([1], dtype=torch.float, requires_grad=False).to(device=device)  # 假标签

    # 初始化回放缓冲区（CycleGAN论文技巧，稳定判别器训练）
    # 存储历史生成的“虚假A风格图像”，避免判别器只看当前生成的样本导致训练波动
    fake_A_buffer = ReplayBuffer()  # 生成器 A 的回放缓冲区
    # 存储历史生成的“虚假B风格图像”，避免判别器只看当前生成的样本导致训练波动
    fake_B_buffer = ReplayBuffer()  # 生成器 B 的回放缓冲区

    # 数据加载器
    transforms_ = transforms.Compose([
        transforms.Resize(int(args.size * 1.12), Image.BICUBIC),
        transforms.RandomCrop(args.size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataloader = DataLoader(
        ImageDataset(os.path.join(args.DOWNLOAD_DIR, args.task_name), 
                     transform=transforms_, model="train"), 
                     batch_size=args.batch_size, shuffle=True, num_workers=8)
    
    # 记录训练迭代步数（每个batch算一步）
    step = 0
    # 获取优化器，网络和损失函数
    opt_G, opt_DA, opt_DB, lr_scheduler_G, lr_scheduler_DA, lr_scheduler_DB = generative_optimizers(args=args)
    netG_A2B, netG_B2A, netD_A, netD_B = etwork_Settings()
    loss_GAN, loss_cycle, loss_identity = loss_functions()

    # 开始训练循环（按epoch遍历，每个epoch遍历所有batch）
    for epoch in range(args.epochs):
        # 遍历每个batch（i是batch索引，batch是当前批次的数据）
        for i, batch in enumerate(dataloader):
            # 准备真实数据
            real_A = torch.tensor(input_A.copy_(batch["A"]), dtype=torch.float).to(device)
            real_B = torch.tensor(input_B.copy_(batch["B"]), dtype=torch.float).to(device)
            # 训练生成器（训练模式）
            netG_A2B.train()
            netG_B2A.train()
            # 清空生成器优化器的梯度（避免上一轮梯度累积）
            opt_G.zero_grad()
            # 恒等损失
            same_B = netG_A2B(real_B)# 把真实B输入A→B生成器，应输出接近B的图像
            loss_identity_B = loss_identity(same_B, real_B) * 5.0 #（论文推荐权重）

            same_A = netG_B2A(real_A)# 把真实A输入B→A生成器，应输出接近A的图像
            loss_identity_A = loss_identity(same_A, real_A) * 5.0 #（论文推荐权重）

            # GAN损失
            fake_A = netG_A2B(real_B)  # A→B生成器输入B，生成"假A"（目标：让D_A认为是真A）
            pred_fake = netD_A(fake_A)  # 判别器A判断"假A"
            loss_GAN_A2B = loss_GAN(pred_fake, label_real) # 生成器A2B的GAN损失（目标：让D_A认为是真A）

            fake_B = netG_B2A(real_A) # B→A生成器输入A，生成"假B"（目标：让D_B认为是真B）
            pred_fake = netD_B(fake_B) # 判别器B判断"假B"
            loss_GAN_B2A = loss_GAN(pred_fake, label_real) # 生成器B2A的GAN损失（目标：让D_B认为是真B）
            
            # cycle loss
            # 计算循环一致性损失（Cycle Loss）：确保"生成→还原"后与原图一致
            recovered_A = netG_B2A(fake_A)
            loss_cycle_ABA = loss_cycle(recovered_A, real_A) * 10.0
            recovered_B = netG_A2B(fake_B)
            loss_cycle_BAB = loss_cycle(recovered_B, real_B) * 10.0
            # 总损失
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()
            # 优化器更新：用梯度更新生成器参数
            opt_G.step()

            # 训练判别器 A
            opt_DA.zero_grad()
            # 真实损失
            pred_real = netD_A(real_A)
            loss_D_real = loss_GAN(pred_real, label_real)
            # 假损失
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = loss_GAN(pred_fake, label_fake)

            # 总损失
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()
            opt_DA.step()
            # 训练判别器 B
            opt_DB.zero_grad()
            # 真实损失
            pred_real = netD_B(real_B)
            loss_D_real = loss_GAN(pred_real, label_real)
            # 假损失
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = loss_GAN(pred_fake, label_fake)
            # 总损失
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()
            opt_DB.step()

            # 更新迭代步数
            step += 1
            
            print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{i+1}/{len(dataloader)}] "
                    f"Loss_G: {loss_G.item():.4f} "
                    f"Loss_G_identity: {(loss_identity_A + loss_identity_B).item():.4f} "
                    f"Loss_G_GAN: {(loss_GAN_A2B + loss_GAN_B2A).item():.4f} "
                    f"Loss_G_cycle: {(loss_cycle_ABA + loss_cycle_BAB).item():.4f} "
                    f"Loss_D_A: {loss_D_A.item():.4f} "
                    f"Loss_D_B: {loss_D_B.item():.4f}")
            
            
            # 日志打印
            logger.info("-" * 150)
            logger.info(f"Epoch [{epoch+1}/{args.epochs}] Batch [{i+1}/{len(dataloader)}] ")
            logger.info(f"生成器总损失[Loss_G]: {loss_G.item():.4f} ")
            logger.info(f"身份损失之和[Loss_G_identity]: {(loss_identity_A + loss_identity_B).item():.4f} ")
            logger.info(f"GAN损失之和[Loss_G_GAN]: {(loss_GAN_A2B + loss_GAN_B2A).item():.4f} ")
            logger.info(f"循环损失之和[Loss_G_cycle]: {(loss_cycle_ABA + loss_cycle_BAB).item():.4f} ")
            logger.info(f"判别器 A 损失[Loss_D_A]: {loss_D_A.item():.4f} ")
            logger.info(f"判别器 B 损失[Loss_D_B]: {loss_D_B.item():.4f}")
            logger.info("-" * 150)
        # 更新学习率
        lr_scheduler_G.step()
        lr_scheduler_DA.step()
        lr_scheduler_DB.step()

        # 每个epoch保存一次模型
        torch.save(netG_A2B.state_dict(), os.path.join(args.MODELS_DIR, f"netG_A2B_epoch_{epoch+1}.pth"))
        torch.save(netG_B2A.state_dict(), os.path.join(args.MODELS_DIR, f"netG_B2A_epoch_{epoch+1}.pth"))
        torch.save(netD_A.state_dict(), os.path.join(args.MODELS_DIR, f"netD_A_epoch_{epoch+1}.pth"))
        torch.save(netD_B.state_dict(), os.path.join(args.MODELS_DIR, f"netD_B_epoch_{epoch+1}.pth"))
        logger.info(f"已保存模型权重到 {args.MODELS_DIR}")
    # 最后保存一次模型
    torch.save(netG_A2B.state_dict(), os.path.join(args.MODELS_DIR, f"netG_A2B_final.pth"))
    torch.save(netG_B2A.state_dict(), os.path.join(args.MODELS_DIR, f"netG_B2A_final.pth"))
    torch.save(netD_A.state_dict(), os.path.join(args.MODELS_DIR, f"netD_A_final.pth"))
    torch.save(netD_B.state_dict(), os.path.join(args.MODELS_DIR, f"netD_B_final.pth"))
    logger.info(f"已保存最终模型权重到 {args.MODELS_DIR}")

    logger.info("-" * 150)
    logger.info(f"训练完成。")
    logger.info("-" * 150)  

# --------------------------------------------------------------------------------------------------------------------- #
# 8.主函数------*[]*
# --------------------------------------------------------------------------------------------------------------------- #
# 运行训练
if __name__ == "__main__":
    train_model()

