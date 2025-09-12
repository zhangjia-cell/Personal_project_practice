import glob
import pickle


# 定义函数用于：读取并解析通过 pickle 序列化的二进制文件，返回反序列化字典对象
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 训练集和测试集的文件列表
train_list = glob.glob('Data/CIFAR10/*/*')
print(train_list)
test_list = glob.glob('Data/CIFAR10/*/*.png')
print(test_list)

# for file in train_list:
#     print(file)
#     file_dict_train = unpickle(file)
#     print(file_dict_train.keys())

# for im_item in train_list:
#     im_label_name = im_item.split('/')[-2]
#     print(im_label_name)