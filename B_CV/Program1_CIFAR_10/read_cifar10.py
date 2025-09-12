# -------------------------------------------------------------------------------------------------------------
# 将下载好的数据进行文件的整理
# -------------------------------------------------------------------------------------------------------------

import os
import cv2
import numpy as np
import glob
import pickle

# 定义函数用于：读取并解析通过 pickle 序列化的二进制文件，返回反序列化字典对象
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 定义训练集和测试集的保存路径
save_train_path = "Data/CIFAR10/Train"
save_test_path = "Data/CIFAR10/Test"

# CIFAR10标签名称
labels_name = [
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

# 训练集和测试集的文件列表
train_list = glob.glob('Data/CIFAR10/data_batch_*')
# print(train_list)
test_list = glob.glob('Data/CIFAR10/test_batch')
# print(test_list)

for file in train_list:
    # print(file)
    file_dict_train = unpickle(file)
    # print(file_dict_train.keys())

    # 训练集
    for im_dix, im_data in enumerate(file_dict_train[b'data']):
        # print(im_dix, im_data)
        im_label = file_dict_train[b'labels'][im_dix]
        im_name = file_dict_train[b'filenames'][im_dix]

        # print(im_label, im_name, im_data)

        im_label_name = labels_name[im_label]
        im_data = np.reshape(im_data, [3, 32, 32])
        im_data = np.transpose(im_data, (1, 2, 0))  # Convert to HWC format

        # cv2.imshow("im_data", cv2.resize((im_data), (200, 200)))
        # cv2.waitKey(0)

        if not os.path.exists("{}/{}".format(save_train_path, im_label_name)):
            os.makedirs("{}/{}".format(save_train_path, im_label_name))
    
        cv2.imwrite("{}/{}/{}".format(save_train_path, im_label_name, im_name.decode("utf-8")), im_data)


for file in test_list:
    # print(file)
    file_dict_test = unpickle(file)
    # print(file_dict.keys())

    # 测试集
    for im_dix, im_data in enumerate(file_dict_test[b'data']):
        # print(im_dix, im_data)
        im_label = file_dict_test[b'labels'][im_dix]
        im_name = file_dict_test[b'filenames'][im_dix]

        # print(im_label, im_name, im_data)

        im_label_name = labels_name[im_label]
        im_data = np.reshape(im_data, [3, 32, 32])
        im_data = np.transpose(im_data, (1, 2, 0))  # Convert to HWC format

        # cv2.imshow("im_data", cv2.resize((im_data), (200, 200)))
        # cv2.waitKey(0)

        if not os.path.exists("{}/{}".format(save_test_path, im_label_name)):
            os.makedirs("{}/{}".format(save_test_path, im_label_name))
        
        if not os.path.exists("{}/{}".format(save_test_path, im_label_name)):
            os.makedirs("{}/{}".format(save_test_path, im_label_name))

        cv2.imwrite("{}/{}/{}".format(save_test_path, im_label_name, im_name.decode("utf-8")), im_data)


