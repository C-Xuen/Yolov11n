import os  
import random  
import shutil

def sample_yolo_dataset(source_images_dir, source_labels_dir, target_images_dir, target_labels_dir, sample_size=1000):  
    # 创建目标文件夹（如果不存在）  
    os.makedirs(target_images_dir, exist_ok=True)  
    os.makedirs(target_labels_dir, exist_ok=True)  

    # 获取图片和标签文件列表  
    image_files = [f for f in os.listdir(source_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]  
    label_files = [f for f in os.listdir(source_labels_dir) if f.endswith('.txt')]  

    # # 确保图片和标签数量一致  
    # assert len(image_files) == len(label_files), "图片和标签数量不匹配"  

    # 从图片文件中随机选择sample_size个文件  
    sampled_images = random.sample(image_files, min(sample_size, len(image_files)))  

    # 移动选中的图片和标签文件到目标文件夹  
    for image_file in sampled_images:  
        label_file = image_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')  
        
        # 移动图片  
        shutil.move(os.path.join(source_images_dir, image_file), target_images_dir)  
        
        # 移动标签，如果标签存在  
        if label_file in label_files:  
            shutil.move(os.path.join(source_labels_dir, label_file), target_labels_dir)  

    print(f"成功随机移动 {len(sampled_images)} 张图片和相应的标签。")  

# 示例用法  
source_images_dir = 'E:/armor_dataset/armor_dataset_v4/images'  # 替换为YOLO图片的路径  
source_labels_dir = 'E:/armor_dataset/armor_dataset_v4/labels'  # 替换为YOLO标签的路径  
target_images_dir = 'E:/armor_dataset/val/images'  # 替换为目标图片的路径  
target_labels_dir = 'E:/armor_dataset/val/labels'  # 替换为目标标签的路径  

sample_yolo_dataset(source_images_dir, source_labels_dir, target_images_dir, target_labels_dir)