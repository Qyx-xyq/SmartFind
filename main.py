import os
from PIL import Image
from YOLO import YOLODetector
from CLIP import CLIPMatcher

def run_pipeline(image_path, text_query):
    print(f"\n=============================================")
    print(f"📸 正在处理图片: {image_path}")
    print(f"🔍 用户查询物品: '{text_query}'")
    print(f"=============================================")

    # 1. 初始化模型（只会加载一次）
    print("--- 初始化模型 ---")
    yolo = YOLODetector(model_path="yolov8n.pt", conf_threshold=0.1)
    clip_matcher = CLIPMatcher(threshold=0.25)

    # 2. YOLO 阶段：提取候选子图
    print("\n--- 阶段 1: YOLO 提取候选区域 ---")
    try:
        crops = yolo.detect_and_crop(image_path)
    except Exception as e:
        print(f"❌ YOLO 处理失败: {e}")
        return

    if not crops:
        print("❌ 结束：YOLO 没有在图片里检测到任何物体。")
        return

    # 3. CLIP 阶段：图文匹配
    print("\n--- 阶段 2: CLIP 图文匹配 ---")
    # 注意：调用队友写的 match_images 方法，传入文字和 YOLO 的切片列表
    result_dict = clip_matcher.match_images(text_query, crops, top_k=3)

    # 4. 解析结果（队友返回的是字典，我们按字典的键取值）
    if result_dict["status"] == "error":
        print(f"❌ CLIP 报错: {result_dict['message']}")
        return

    print(f"判定结论: {result_dict['message']}")
    print(f"最高得分: {result_dict['best_score']}")

    # 5. 如果有找到的目标，把得分最高的那张切片保存下来
    if result_dict["is_found"]:
        top_results = result_dict["top_k_results"]
        if top_results:
            # top_results 是一个列表，里面是 (PIL图片, 得分) 的元组
            best_image = top_results[0][0] 
            
            # 保存这张图，方便你们写报告和展示
            save_dir = "data/outputs"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "found_target.jpg")
            best_image.save(save_path)
            print(f"✨ 最匹配的切片已保存至: {save_path}")

    print("=============================================\n")


if __name__ == "__main__":
    # 找一张真实的图片测试（确保 data 文件夹下有这张图）
    # 如果没有，你可以先去网上下载一张带背包的图，命名为 test.jpg 放进 data 文件夹
    test_img = "data/test.jpg" 
    test_query = "a black backpack"
    
    # 运行流水线
    run_pipeline(test_img, test_query)