import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import nncf

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")  # 确保图像为 RGB 格式

        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]  # 返回图像和文件名作为标签

if __name__ == '_main_':   

    # 使用自定义数据集
    data_path = 'E:\\armor_dataset\\armor_dataset_v4\\images'
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    custom_dataset = CustomImageDataset(root_dir=data_path, transform=transform)

    # 创建数据加载器
    data_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True, num_workers=4)

    # 迭代数据加载器
    for images, filenames in data_loader:
        # 处理图像和文件名
        print(filenames)  # 打印文件名
        # 这里可以添加对图像的处理逻辑

    # 定义转换函数
    def transform_fn(data_item):
        images, _ = data_item
        return images

    # 创建 nncf.Dataset 实例
    calibration_dataset = nncf.Dataset(data_loader, transform_fn)