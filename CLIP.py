"""
CLIP模块，图文语义匹配，为最终系统提供决策依据
负责人：黄涵
"""
import torch
import clip
from PIL import Image

class CLIPMatcher:
    def __init__(self, model_name="ViT-B/32", threshold=0.25):
        """
        初始化 CLIP 匹配器
        :param model_name: CLIP 模型名称，默认使用最轻量的 ViT-B/32 模型
        :param threshold: 相似度阈值。超过此阈值判定为找到目标物品（建议根据实际测试在 0.2~0.3 之间调整）
        """
        print(f"正在加载 CLIP 模型: {model_name} ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.threshold = threshold
        print(f"CLIP 模型加载完成，当前设备: {self.device}")

    def match_images(self, text_query: str, candidate_images: list, top_k: int = 5):
        """
        核心功能：将文本描述与 YOLO 裁剪出的候选图像列表进行匹配
        :param text_query: 用户输入的物品描述，如 "a black backpack"
        :param candidate_images: 候选子图列表 (list of PIL.Image)
        :param top_k: 返回得分最高的前 K 个结果
        :return: 字典，包含是否找到目标、最高得分、Top-K详细结果
        """
        # 1. 异常处理：输入为空
        if not text_query or not text_query.strip():
            return {"status": "error", "message": "输入文本为空，请重新输入"}
        if not candidate_images:
            return {"status": "error", "message": "候选图像列表为空"}

        # 限制 top_k 不能超过候选图片总数
        top_k = min(top_k, len(candidate_images))

        # 2. 文本特征提取
        with torch.no_grad():
            text_input = clip.tokenize([text_query]).to(self.device)
            text_features = self.model.encode_text(text_input)
            # L2 归一化
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # 3. 图像特征批量提取 (Batch Inference 加速)
            # 将所有的 PIL Image 堆叠成一个 Batch Tensor
            image_inputs = torch.stack([self.preprocess(img) for img in candidate_images]).to(self.device)
            image_features = self.model.encode_image(image_inputs)
            # L2 归一化
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # 4. 计算相似度 (余弦相似度等价于归一化后的点积)
            # 乘以 100 是为了放大差异，结果 shape 为 (1, N)
            similarity_scores = (100.0 * text_features @ image_features.T).squeeze(0)
            
            # 转化为 Python 列表便于处理
            scores = similarity_scores.cpu().numpy().tolist()

        # 5. 结果排序与聚合
        # 将 (图片, 得分) 组合，并按得分从高到低排序
        results = list(zip(candidate_images, scores))
        results.sort(key=lambda x: x[1], reverse=True)

        # 提取 Top-K
        top_results = results[:top_k]
        best_score = top_results[0][1] if top_results else 0.0

        # 6. 阈值判定
        is_found = best_score >= self.threshold

        return {
            "status": "success",
            "is_found": is_found,
            "best_score": best_score,
            "top_k_results": top_results, # 包含 (PIL.Image, score) 的列表
            "message": "✅ 找到目标！" if is_found else "❌ 未找到目标（相似度未达到阈值）"
        }

# ================= 本地测试 =================
if __name__ == "__main__":
    # 模拟 YOLO 的输出（这里为了不依赖外部图片文件，直接生成两张测试图）
    # 实际使用时，你会直接把同学写的 YOLODetector 返回的 crops_pil 传进来
    img_blue = Image.new('RGB', (224, 224), color='blue')
    img_red = Image.new('RGB', (224, 224), color='red')
    mock_yolo_crops = [img_blue, img_red]

    # 实例化匹配器
    matcher = CLIPMatcher(threshold=0.25)

    # 测试：找蓝色的东西
    query = "a blue picture"
    print(f"\n--- 正在匹配: '{query}' ---")
    result = matcher.match_images(query, mock_yolo_crops, top_k=2)
    
    print(f"判定结果: {result['message']}")
    print(f"最高得分: {result['best_score']:.4f}")
    
    for idx, (crop_img, score) in enumerate(result['top_k_results']):
        print(f"  Top {idx+1}: 得分 {score:.4f}")