from ultralytics import YOLO
import cv2

# 1. 加载官方预训练模型（首次运行会自动下载 yolov8n.pt）
model = YOLO("yolov8n.pt")

# 2. 对示例图片进行推理
results = model("demo_pipeline/data/bus.jpg")

# 3. 处理检测结果并裁剪
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        crop_img = result.orig_img[y1:y2, x1:x2]
        if crop_img.size > 0:
            cv2.imwrite(f"demo_pipeline/data/crop_{i}.jpg", crop_img)
            print(f"已保存 crop_{i}.jpg")