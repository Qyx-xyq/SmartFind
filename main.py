import os
import glob
from YOLO import YOLODetector
from CLIP import CLIPMatcher

def batch_process_folder(folder_path, text_query):
    print(f"\n📂 正在批量处理文件夹: {folder_path}")
    print(f"🔍 寻找目标: '{text_query}'")

    # 1. 初始化模型
    yolo = YOLODetector(model_path="models/yolov8n.pt", conf_threshold=0.1)
    clip_matcher = CLIPMatcher(threshold=0.25)

    # 2. 获取文件夹内所有图片
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not image_paths:
        print("❌ 文件夹里没有找到图片！")
        return

    # 创建输出文件夹，用于保存你所有的裁剪图
    save_dir = "data/outputs"
    os.makedirs(save_dir, exist_ok=True)

    # 3. 遍历所有图片
    for img_path in image_paths:
        print(f"\n📸 处理图片: {os.path.basename(img_path)}")
        
        # YOLO 切图
        crops = yolo.detect_and_crop(img_path)
        if not crops:
            print("  - YOLO 未检测到任何物体")
            continue
            
        print(f"  - YOLO 切出了 {len(crops)} 个小图，正在交给 CLIP 打分...")
        
        # CLIP 匹配（这里把 top_k 设大一点，比如 100，相当于返回所有结果）
        res = clip_matcher.match_images(text_query, crops, top_k=100)
        
        if res["status"] == "error":
            print(f"  - CLIP 报错: {res['message']}")
            continue
            
        # 4. 保存所有切出来的小图，并在文件名上打上分数
        top_results = res["top_k_results"]
        for i, (crop_img, score) in enumerate(top_results):
            # 把得分乘100取整放在文件名里，方便你直接按分数看
            score_pct = int(score * 100)
            save_name = f"{os.path.basename(img_path).split('.')[0]}_crop{i}_score_{score_pct}.jpg"
            crop_img.save(os.path.join(save_dir, save_name))
            
        print(f"  ✅ 处理完毕，最高分: {res['best_score']:.3f}。所有切片已保存至 {save_dir}")

if __name__ == "__main__":
    # 确保你把测试图放在 data 文件夹下
    batch_process_folder("web/test_images", "red backpack")