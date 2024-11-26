import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class MyDataset(Dataset):
    def __init__(self, file_list, data_root, transform_type):
        """
        Args:
            file_list (str): 存储图像路径和标签的文件（如 TR.txt）
            data_root (str): 数据集根目录
            transform_type (str): 应用于图像的变换
        """
        self.data_root = data_root
        self.data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
        self.transform = self.data_transform[transform_type]
        # 读取文件列表并解析图像路径和标签
        with open(file_list, 'r') as f:
            self.image_paths = f.readlines()

        # 提取标签
        self.labels = []
        for line in self.image_paths:
            # 假设每行格式为 "/<label>/filename.jpg"
            label = int(line.strip().split('/')[1])-1  # 获取路径中的类别标签（即第二部分）
            self.labels.append(label)

        # 标签转换为NumPy数组
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 获取图像路径
        img_path = os.path.join(self.data_root, self.image_paths[idx].strip().lstrip('/'))
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        # 应用数据转换（例如 resize, normalize等）
        if self.transform:
            image = self.transform(image)

        return image, label



def create_food_label_dict(file_path):
    """
    读取文件，生成标签与食物名称的映射字典
    Args:
        file_path (str): FoodList.txt 的路径
    Returns:
        dict: 标签与食物名称的映射字典
    """
    food_label_dict = {}

    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        food_list = file.readlines()

    # 创建标签和食物名称的映射字典
    for idx, food in enumerate(food_list):
        food_name = food.strip()  # 去掉每行的换行符
        food_label_dict[idx] = food_name

    return food_label_dict
