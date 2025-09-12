from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import glob

# CIFAR10标签名称
label_name = [
    'airplane', 
    'automobile', 
    'bird', 
    'cat', 
    'deer', 
    'dog', 
    'frog', 
    'horse', 
    'ship', 
    'truck'
]

# 创建标签字典，将标签名称映射到索引
label_dict = {}
for idx, name in enumerate(label_name):
    label_dict[name] = idx

# 加载图片文件并将其转换为RGB格式
def default_loader(path):
    return Image.open(path).convert("RGB")

# # 定义数据增强
# train_transform = transforms.Compose([
#     # 随机剪切
#     transforms.RandomResizedCrop((28, 28)),
#     # 随机水平翻转
#     transforms.RandomHorizontalFlip(),
#     # 随机垂直翻转
#     transforms.RandomVerticalFlip(),
#     # 随机旋转
#     transforms.RandomRotation(90),
#     # 随机转换为灰度图
#     transforms.RandomGrayscale(p=0.1),
#     # 颜色信息增强
#     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
#     # 转换为Tensor
#     transforms.ToTensor(),
#     # 归一化
#     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# 简化数据增强
train_transform = transforms.Compose([
    transforms.RandomCrop(28),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# 自定义数据集处理器
class MyDataset(Dataset):
    def __init__(self, im_list, transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        imgs = []

        for im_item in im_list:
            im_label_name = im_item.split("\\")[-2]
            imgs.append([im_item, label_dict[im_label_name]])

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        im_path, im_label = self.imgs[index]
        im_data = self.loader(im_path)
        if self.transform is not None:
            im_data = self.transform(im_data)
        return im_data, im_label

    def __len__(self):
        return len(self.imgs)

# 获取训练和测试数据集
im_train_list = glob.glob("Data\CIFAR10\Train\*\*")
# im_train_list = glob.glob("F:\Code_Study\Program1_CIFAR_10\Data\CIFAR10\Test\*\*.png")
im_test_list = glob.glob("Data\CIFAR10\Test\*\*")

 
# 创建数据集和数据加载器[训练集和测试集]
train_dataset = MyDataset(im_train_list, transform=train_transform)
test_dataset = MyDataset(im_test_list, transform=transforms.ToTensor())


train_data_loader = DataLoader(
    # 训练集
    train_dataset, 
    # 每批加载的样本数
    batch_size=6, 
    # 是否在每个 epoch 开始时打乱数据顺序
    shuffle=True, 
    # 用于数据加载的子进程数（默认值：0，即主进程加载）。
    # 设置为大于 0 的值可开启多进程并行加载，加速数据读取（通常设为 CPU 核心数或其倍数）。
    num_workers=4, 
    # 是否将加载的数据复制到 CUDA 固定内存中
    # 若使用 GPU 训练，设为 True 可加速数据从 CPU 到 GPU 的传输。
    pin_memory=True
)
test_data_loader = DataLoader(
    test_dataset,
    batch_size=6,
    shuffle=False,
    num_workers=4
)

# print("num of train", len(train_dataset))
# print("num of test", len(test_dataset))


