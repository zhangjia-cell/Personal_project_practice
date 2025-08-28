import os
import glob
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms



# --------------------------------------------------------------------------------------------------------------------- #
# 1.项目文件设置------*[需要修改]*
# --------------------------------------------------------------------------------------------------------------------- #
Program_folder = "C_Generative\GAN1_Image_Style_Transfer"
Data_folder = "Data"
Temp_folder = os.path.join(Program_folder, Data_folder)

if not os.path.exists(Temp_folder):
    os.makedirs(Temp_folder, exist_ok=True)
    print(f"已创建文件夹: {Temp_folder}")
else:
    print(f"文件夹已存在: {Temp_folder}")

# --------------------------------------------------------------------------------------------------------------------- #
# 2.数据集定义------*[处理数据]*
# --------------------------------------------------------------------------------------------------------------------- #
class ImageDataset(Dataset):
    def __init__(self, root="", transform=None, model="train"):
        super().__init__()
        self.transform = transforms.Compose(transform)

        self.pathA = os.path.join(root, model, "A/*")
        self.pathB = os.path.join(root, model, "B/*")

        self.list_A = glob.glob(self.pathA)
        self.list_B = glob.glob(self.pathB)

    def __getitem__(self, index):
        im_pathA = self.list_A[index % len(self.list_A)]
        im_pathB = random.choice(self.list_B)

        im_A= Image.open(im_pathA)
        im_B= Image.open(im_pathB)

        item_A = self.transform(im_A)
        item_B = self.transform(im_B)

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.list_A), len(self.list_B))

if __name__ == "__main__":
    from torch.utils.data import Dataloader

    root = "datasets/apple2orange"

    transform_ = [transforms.Resize(256, Image.BICUBIC)]
    dataloader = Dataloader(ImageDataset(root, transform_, "train"), batch_size=1, shuffle=True, num_workers=1)

    for i, batch in enumerate(dataloader):
        print(i)
        print(batch["A"].shape)














