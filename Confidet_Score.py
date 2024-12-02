import numpy as np  
import cv2  
import onnxruntime as ort  

 # 置信度

def prepare_input(frame):  
    frame_resized = cv2.resize(frame, (224, 224))  
    frame_normalized = frame_resized / 255.0  
    input_tensor = frame_normalized.transpose((2, 0, 1)).astype(np.float32)  
    input_tensor = input_tensor[np.newaxis, :]  
    return input_tensor  

def load_model(model_path):  
    session = ort.InferenceSession(model_path)  
    return session  

def inference(session, input_tensor):  
    output = session.run(['output0'], {'images': input_tensor})  
    return output[0]  

def softmax(x):  
    """计算 x 中每组分数的 softmax 值。"""
    exp_x = np.exp(x - np.max(x))  # 为了数值稳定性  
    return exp_x / exp_x.sum(axis=-1, keepdims=True)  

def process_output(output_tensor):  
    output_shape = output_tensor.shape  
    print(f"输出张量形状： {output_shape}")  

    if len(output_shape) == 3:  
        # 将1029维的特征对每个类别求平均或选择最大  
        mean_values = np.mean(output_tensor, axis=2)  # 对最后一维求平均  

        # 计算softmax以获得置信度  
        probabilities = softmax(mean_values[0])  # softmax化  
        predicted_class = np.argmax(probabilities)  # 获取最大值索引  
        confidence_score = probabilities[predicted_class]  # 获取对应的置信度  
    else:  
        raise ValueError("Unexpected output shape from the model, expected 3 dimensions.")  

    return predicted_class, confidence_score
if __name__ == "__main__":  
    video_path = 'E:\\ros_ws\src\\rm_yolo\\rm_yolo_aim\\test\\test.mp4'  # 输入视频路径  

    # 启动虚拟环境，输入：yolo export model='你的.pt' format=onnx imgsz=224
    model_path = 'E:\\ros_ws\src\\rm_yolo\\rm_yolo_aim\\rm_yolo_aim\models\\best.onnx'          # ONNX模型路径  
    # E:\ros_ws\src\rm_yolo\rm_yolo_aim\rm_yolo_aim\models\best.onnx
    # yolov11n的：E:\Yolov11n\\runs\detect\\train16\weights\\best.onnx
    
    session = load_model(model_path)  
    cap = cv2.VideoCapture(video_path)  

    if not cap.isOpened():  
        print("Error: Could not open video.")  
        exit()  

    while True:  
        ret, frame = cap.read()  
        if not ret:  
            break  

        input_tensor = prepare_input(frame)  
        output_tensor = inference(session, input_tensor)  
        
        predicted_class, confidence_score = process_output(output_tensor)  
        print(f"预测等级： {predicted_class}, 置信度: {confidence_score:.4f}")  

        cv2.imshow('Frame', frame)  
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break  

    cap.release()  
    cv2.destroyAllWindows()
