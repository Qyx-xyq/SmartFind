import torch
import clip
import cv2
from PIL import Image
from ultralytics import YOLO

def init_models():
    """初始化加载两个模型，避免在循环中重复加载"""
    print("正在加载 YOLO 模型...")
    yolo_model = YOLO("yolov8n.pt")
    
    print("正在加载 CLIP 模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
    return yolo_model, clip_model, preprocess, device

def find_object_in_image(image_path, text_query, yolo_model, clip_model, preprocess, device, threshold=0.25):
    """
    核心流水线：YOLO 切片 -> 格式转换 -> CLIP 打分
    """
    print(f"\n开始处理图片: {image_path}")
    print(f"寻找目标: '{text_query}'")

    # ---------------- 1. YOLO 阶段：找候选框并切图 ----------------
    results = yolo_model(image_path)
    crops_pil = [] # 存放在内存中的 PIL 图像，供 CLIP 使用
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        # BGR 转 RGB 用于展示和 CLIP 识别
        orig_img_rgb = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            # 防止切片越界
            x1, y1 = max(0, x1), max(0, y1)
            
            crop_bgr = result.orig_img[y1:y2, x1:x2]
            if crop_bgr.size > 0:
                # 转换颜色通道 BGR -> RGB
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                # 转为 PIL 格式
                crop_pil = Image.fromarray(crop_rgb)
                crops_pil.append(crop_pil)
                
                # (可选) 如果想保存下来看效果，取消下面两行的注释
                crop_pil.save(f"debug_crop_{i}.jpg")
    
    print(f"YOLO 抠出了 {len(crops_pil)} 个候选子图。")
    
    if len(crops_pil) == 0:
        return "YOLO 未检测到任何物体", 0.0

    # ---------------- 2. CLIP 阶段：图文匹配 ----------------
    # 文本特征提取
    text_input = clip.tokenize([text_query]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_input)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        best_score = -1.0
        best_crop = None
        
        # 逐个计算候选子图与文本的相似度
        for crop_pil in crops_pil:
            # 图像预处理
            image_input = preprocess(crop_pil).unsqueeze(0).to(device)
            image_features = clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # 计算余弦相似度 (点积)
            similarity = (image_features @ text_features.T).item()
            
            # 记录最高分
            if similarity > best_score:
                best_score = similarity
                best_crop = crop_pil

    # ---------------- 3. 结果判定 ----------------
    print(f"CLIP 计算的最高相似度得分为: {best_score:.4f}")
    
    if best_score >= threshold:
        # 你可以把 best_crop 返回，或者存下来展示给用户
        return "✅ 找到目标！", best_score
    else:
        return "❌ 未找到目标（相似度未达到阈值）", best_score

# ================= 测试运行 =================
if __name__ == "__main__":
    # 初始化模型
    yolo, clip_m, prep, dev = init_models()
    
    # 测试图片（这里用官方测试图，里面有一辆公交车和几个人）
    test_image = "bus.jpg"
    
    # 测试查询 1：找“人”
    status, score = find_object_in_image(test_image, "a photo of a person", yolo, clip_m, prep, dev)
    print(f"判定结果: {status}, 得分: {score:.4f}")
    
    # 测试查询 2：找“公交车”
    status, score = find_object_in_image(test_image, "a photo of a bus", yolo, clip_m, prep, dev)
    print(f"判定结果: {status}, 得分: {score:.4f}")
    
    # 测试查询 3：找“猫”（图里没有，得分应该会很低）
    status, score = find_object_in_image(test_image, "a photo of a cat", yolo, clip_m, prep, dev)
    print(f"判定结果: {status}, 得分: {score:.4f}")