# train_yolo.py
from ultralytics import YOLO

def main():
    # 1. 加载预训练权重
    model = YOLO("yolov8n.pt")
    
    # 2. 开始微调（前提：你必须按照 YOLO 格式准备好数据集）
    model.train(
        data="data/my_dataset/data.yaml",  # 你的数据集配置
        epochs=50,
        imgsz=640,
        batch=8,
        device=0, # 用 GPU
        name="smartfind_custom",
        # 小目标优化的训练参数
        mosaic=1.0, 
        close_mosaic=10,
        hsv_v=0.5
    )

if __name__ == "__main__":
    main()