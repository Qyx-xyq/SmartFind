"""
YOLO模块，物体识别裁切，提高物品识别的准确性
秦宇霄负责
"""
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="model/yolov8n.pt", conf_threshold=0.15, min_size=32):
        """
        初始化 YOLO 检测器
        :param model_path: 模型路径，如 'yolov8n.pt' 或微调后的 'runs/.../best.pt'
        :param conf_threshold: 置信度阈值。监控场景建议调低（如 0.15），宁可错杀不可放过
        :param min_size: 最小裁剪尺寸（像素）。太小的框（如 10x10）没意义，直接过滤
        """
        print(f"正在加载 YOLO 模型: {model_path} ...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.min_size = min_size

    def detect_and_crop(self, image_path):
        """
        核心功能：对一张大图进行检测，并裁剪出候选子图
        :param image_path: 输入大图的路径（或 numpy 数组）
        :return: 包含裁剪后 PIL.Image 对象的列表
        """
        # 1. 执行推理（设置 verbose=False 关闭控制台刷屏）
        results = self.model(image_path, conf=self.conf_threshold, verbose=False)
        
        crops_pil = []
        
        for result in results:
            # 获取边界框坐标
            boxes = result.boxes.xyxy.cpu().numpy()
            
            # 获取原始图像（YOLO 默认读入是 BGR numpy 格式）
            orig_img_bgr = result.orig_img
            
            # 获取图像的宽高，用于边界安全校验
            img_h, img_w = orig_img_bgr.shape[:2]

            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                
                # 2. 边界安全校验（防止坐标越界）
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)
                
                w = x2 - x1
                h = y2 - y1

                # 3. 尺寸过滤（过滤掉极小、极细长的无效框）
                if w < self.min_size or h < self.min_size:
                    continue

                # 4. 裁剪图像
                crop_bgr = orig_img_bgr[y1:y2, x1:x2]
                
                # 5. 颜色通道转换 BGR -> RGB (关键：CLIP 需要 RGB 格式)
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                
                # 6. 转成 PIL Image 并加入列表
                crop_pil = Image.fromarray(crop_rgb)
                crops_pil.append(crop_pil)

        print(f"[YOLO] 处理完成，共提取出 {len(crops_pil)} 个候选子图。")
        return crops_pil

# ================= 本地测试 =================
if __name__ == "__main__":
    # 实例化检测器（调整 conf_threshold 可以控制框的多少）
    detector = YOLODetector(model_path="models/yolov8n.pt", conf_threshold=0.1)
    
    # 测试图片
    test_img = "data/bus.jpg"
    
    # 获取裁剪结果
    cropped_images = detector.detect_and_crop(test_img)
    
    # 把切出来的第一张图保存到本地看看效果
    if cropped_images:
        cropped_images[0].save("YOLOtest/test_crop_result1.jpg")
        print("已保存第一张切片为 test_crop_result1.jpg")
        cropped_images[1].save("YOLOtest/test_crop_result2.jpg")
        print("已保存第二张切片为 test_crop_result2.jpg")