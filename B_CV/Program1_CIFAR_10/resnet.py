import torch
import torch.nn as nn
import torch.nn.functional as F

# ResNet 的模块
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        # 主干层
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        # 跳跃连接
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out1 = self.layer(x)
        out2 = self.shortcut(x)
        out = out1 + out2
        out = F.relu(out)
        return out

class ResNet_base(nn.Module):

    def make_layer(self, block, out_channels, stride=1, num_blocks=1):
        layers_list = []
        for i in range(num_blocks):
            if i == 0:
                in_stride = stride
            else:
                in_stride = 1
            layers_list.append(block(self.in_channels, out_channels, in_stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers_list)

    def __init__(self):
        super(ResNet_base, self).__init__()
        self.in_channels = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False)
        )
        self.layer1 = self.make_layer(ResBlock, 64, stride=2, num_blocks=3)
        self.layer2 = self.make_layer(ResBlock, 128, stride=2, num_blocks=3)
        self.layer3 = self.make_layer(ResBlock, 256, stride=2, num_blocks=3)
        self.layer4 = self.make_layer(ResBlock, 512, stride=2, num_blocks=3)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
def ResNet():
    return ResNet_base()
