import torch
import torch.nn as nn
import torchvision
from vggnet import VGGNet
from resnet import ResNet
from load_cifar10 import train_data_loader, test_data_loader
import os
import tensorboardX


def main():
    # 定义训练的环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_num = 200
    lr = 0.01

    # net = VGGNet().to(device)
    net = ResNet().to(device)

    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) # 待优化的参数，学习率， 动量参数， 权重衰减(L2正则化)

    # 有关学习率调整的方法
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.9)

    if not os.path.exists("logs"):
        os.makedirs("logs")
    writer = tensorboardX.SummaryWriter("logs")

    step_n = 0
    # 定义训练过程
    for epoch in range(epoch_num):
        print("-------Epoch : {} -------".format(epoch + 1))
        # 告知当前是训练过程
        # 对数据进行遍历
        for i, data in enumerate(train_data_loader):
            
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()  # 清除梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            _, pred = torch.max(outputs.data, dim=1)

            correct = pred.eq(labels.data).cpu().sum()
            # print("epoch is ", epoch)
            # print("Train Step:", i, "Loss is:", loss.item(), "mini-batch correct is:", 100.0 * correct / train_data_loader.batch_size)
            writer.add_scalar("train loss", loss.item(), global_step=step_n)
            writer.add_scalar("train accuracy", 100.0 * correct.item() / train_data_loader.batch_size, global_step=step_n)
            
            im = torchvision.utils.make_grid(inputs)
            writer.add_image("train images", im, global_step=step_n)

            step_n += 1
        if not os.path.exists("Models"):
            os.makedirs("Models")
        torch.save(net.state_dict(), "Models/{}.pth".format(epoch + 1))
        scheduler.step()
        print("Train learning rate is:", optimizer.state_dict()['param_groups'][0]['lr'])

        sum_loss = 0.0
        sum_correct = 0.0
        for i, data in enumerate(test_data_loader):
            # 告知当前是测试过程
            net.eval()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = loss_function(outputs, labels)

            _, pred = torch.max(outputs.data, dim=1)
            correct = pred.eq(labels.data).cpu().sum()

            sum_loss += loss.item()
            sum_correct += correct.item()
            im = torchvision.utils.make_grid(inputs)

            writer.add_image("test images", im, global_step=epoch + 1)

        test_loss = sum_loss * 1.0 / len(test_data_loader)
        test_correct = sum_correct * 1.0 / len(test_data_loader.dataset) / test_data_loader.batch_size
        
        writer.add_scalar("test loss", test_loss, global_step=epoch + 1)
        writer.add_scalar("test accuracy", 100.0 * test_correct, global_step=epoch + 1)
        
        print("Test Loss is:", sum_loss / len(test_data_loader), "Test Accuracy is:", 100.0 * sum_correct / len(test_data_loader.dataset))

    # 关闭记录
    writer.close()

if __name__ == '__main__':
    main()