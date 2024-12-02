from ultralytics import YOLO
import os
 
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # 设置环境变量（不推荐）

# Load a model
def main():
    model = YOLO("yolo11n.yaml")  # build a new model from YAML
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

    results = model.train(
        data="armor.yaml", 
        epochs = 50, 
        imgsz = 640, 
        device = "0")

    # 评估模型
    results = model.val()
    metrics = results.box.map  # 获取mAP@0.5:0.95
    metrics = results.box.map50  # 获取mAP@0.5

if __name__ == '__main__':
    main()