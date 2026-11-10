# ==========================================
# 解决 Mac 系统下 OpenMP 库冲突导致的崩溃
# 必须放在所有 import 的最前面！
# ==========================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

"""
CLIP 图文语义匹配模块 (M5)
负责人：黄涵

说明：
1. 采用全局单例加载，确保整个系统运行期间模型只加载一次，极大降低内存占用和响应时间。
2. 采用 Batch Inference (批处理) 技术，无论 YOLO 切出多少张图，一次性送入模型，速度极快。
3. 对外提供 CLIPMatcher 类，方便 UI 层直接实例化调用。
"""

import torch
import clip
from PIL import Image

# ================= 全局模型单例加载 =================
# 保证整个系统进程只加载一次模型，极大地提高了多次查询时的速度
_GLOBAL_MODEL = None
_GLOBAL_PREPROCESS = None
_GLOBAL_DEVICE = None

def load_clip_model(model_name="ViT-B/32"):
    """加载 CLIP 模型（单例模式），返回 (模型, 预处理, 设备)。"""
    global _GLOBAL_MODEL, _GLOBAL_PREPROCESS, _GLOBAL_DEVICE
    if _GLOBAL_MODEL is not None:
        return _GLOBAL_MODEL, _GLOBAL_PREPROCESS, _GLOBAL_DEVICE
    
    _GLOBAL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在加载 CLIP 模型: {model_name} (设备: {_GLOBAL_DEVICE})...")
    _GLOBAL_MODEL, _GLOBAL_PREPROCESS = clip.load(model_name, device=_GLOBAL_DEVICE)
    print("CLIP 模型加载完成！")
    return _GLOBAL_MODEL, _GLOBAL_PREPROCESS, _GLOBAL_DEVICE


# ================= 核心匹配类 =================
class CLIPMatcher:
    def __init__(self, model_name="ViT-B/32", threshold=0.25):
        """
        初始化 CLIP 匹配器
        :param model_name: CLIP 模型名称，默认最轻量的 ViT-B/32
        :param threshold: 相似度判定阈值，超过此值判定为找到目标 (建议 0.2~0.3)
        """
        # 调用全局单例，如果别处已经加载过了，这里就是秒开
        self.model, self.preprocess, self.device = load_clip_model(model_name)
        self.threshold = threshold

    def match_images(self, text_query: str, candidate_images: list, top_k: int = 5):
        """
        核心功能：将文本描述与 YOLO 裁剪出的候选图像列表进行匹配
        :param text_query: 用户输入的物品描述，如 "a black backpack"
        :param candidate_images: 候选子图列表 (list of PIL.Image)
        :param top_k: 返回得分最高的前 K 个结果
        :return: 字典，包含是否找到目标、最高得分、Top-K详细结果
        """
        # —— 异常处理 1：输入文本为空 ——
        if text_query is None or not str(text_query).strip():
            return {
                "status": "error", 
                "is_found": False,
                "message": "输入文本为空，请重新输入物品描述",
                "best_score": 0.0,
                "top_k_results": []
            }
        
        # —— 异常处理 2：没有收到 YOLO 候选子图 ——
        if not candidate_images:
            return {
                "status": "error", 
                "is_found": False,
                "message": "未收到候选子图（YOLO 未检测到任何物体）",
                "best_score": 0.0,
                "top_k_results": []
            }

        # 限制 top_k 不能超过候选图片总数
        top_k = min(top_k, len(candidate_images))
        text_query = str(text_query).strip()

        try:
            # 1. 文本特征提取
            with torch.no_grad():
                text_input = clip.tokenize([text_query]).to(self.device)
                text_features = self.model.encode_text(text_input)
                # L2 归一化
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # 2. 图像特征批量提取 (Batch Inference 加速)
                # 将所有的 PIL Image 堆叠成一个 Batch Tensor，一次性送入模型
                image_inputs = torch.stack([self.preprocess(img) for img in candidate_images]).to(self.device)
                image_features = self.model.encode_image(image_inputs)
                # L2 归一化
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # 3. 计算相似度 (余弦相似度等价于归一化后的点积)
                # 乘以 100 是为了放大差异，结果 shape 为 (1, N)
                similarity_scores = (100.0 * text_features @ image_features.T).squeeze(0)
                scores = similarity_scores.cpu().numpy().tolist()

            # 4. 结果排序与聚合
            # 将 (图片, 得分) 组合，并按得分从高到低排序
            results = list(zip(candidate_images, scores))
            results.sort(key=lambda x: x[1], reverse=True)

            # 提取 Top-K
            top_results = results[:top_k]
            best_score = top_results[0][1] if top_results else 0.0

            # 5. 阈值判定
            is_found = best_score >= self.threshold

            return {
                "status": "success",
                "is_found": is_found,
                "best_score": round(best_score, 4),
                "top_k_results": top_results, # 包含 (PIL.Image, score) 的列表
                "message": "✅ 找到目标！" if is_found else "❌ 未找到目标（相似度未达到阈值）"
            }
            
        except Exception as e:
            # 捕获模型推理时的意外错误（比如图片损坏等）
            return {
                "status": "error", 
                "is_found": False,
                "message": f"图像编码或匹配失败: {str(e)}",
                "best_score": 0.0,
                "top_k_results": []
            }

# ================= 本地独立测试 =================
if __name__ == "__main__":
    # 模拟 YOLO 的输出（纯色图测试，不需要外部文件）
    img_blue = Image.new('RGB', (224, 224), color='blue')
    img_red = Image.new('RGB', (224, 224), color='red')
    mock_yolo_crops = [img_blue, img_red]

    # 实例化匹配器（此时会打印“正在加载 CLIP 模型...”）
    matcher = CLIPMatcher(threshold=0.20)

    print("\n--- 第一次查询：找蓝色图 ---")
    result = matcher.match_images("a blue picture", mock_yolo_crops, top_k=2)
    print(f"判定结果: {result['message']}")
    print(f"最高得分: {result['best_score']:.4f}")
    for idx, (img, score) in enumerate(result['top_k_results']):
        print(f"  Top {idx+1}: 得分 {score:.4f}")

    print("\n--- 第二次查询：找红色图 (验证单例加载) ---")
    # 注意：这里不会再次打印“正在加载 CLIP 模型...”，证明单例机制生效，瞬间完成
    result2 = matcher.match_images("a red picture", mock_yolo_crops, top_k=2)
    print(f"判定结果: {result2['message']}")
    print(f"最高得分: {result2['best_score']:.4f}")