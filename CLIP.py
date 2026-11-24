# ==========================================
# 解决 Mac 系统下 OpenMP 库冲突导致的崩溃
# 必须放在所有 import 的最前面！
# ==========================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

"""
CLIP 图文语义匹配模块 (M5)
负责人：黄涵

==================== 最新版更新说明 ====================
1. 【关键修复】相似度口径统一为 0~1 的余弦相似度：
   旧版乘了 100（得分 20~40），导致前端 score*100 显示成 2500%，
   且 0.25 阈值形同虚设（任何得分都远超它）。
   现在 score 直接是 0~1 的相似度，前端 *100 显示百分比、0.25 阈值均正确。
2. 单例加载 + 线程锁：Flask/Streamlit 多线程下也只加载一次模型。
3. 批处理推理：所有候选子图一次编码，速度快。
4. 新增 encode_image_feature()：供失物登记（item_register.py）提取图片特征复用。
5. 完整异常处理 + 输入校验，兼容原有全部调用方式。

对外接口保持不变：
    load_clip_model(model_name) -> (model, preprocess, device)
    CLIPMatcher(model_name, threshold).match_images(text, [PIL.Image], top_k) -> dict
"""

import threading
from typing import List

import numpy as np
import torch
import clip
from PIL import Image

# ================= 全局模型单例（线程安全） =================
_MODEL = None
_PREPROCESS = None
_DEVICE = None
_MODEL_LOCK = threading.Lock()


def load_clip_model(model_name: str = "ViT-B/32"):
    """加载 CLIP 模型（线程安全单例），返回 (模型, 预处理, 设备)。"""
    global _MODEL, _PREPROCESS, _DEVICE
    if _MODEL is not None:
        return _MODEL, _PREPROCESS, _DEVICE
    with _MODEL_LOCK:
        if _MODEL is None:  # 双重检查，防止并发重复加载
            _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"正在加载 CLIP 模型: {model_name} (设备: {_DEVICE})...")
            _MODEL, _PREPROCESS = clip.load(model_name, device=_DEVICE)
            _MODEL.eval()
            print("CLIP 模型加载完成！")
    return _MODEL, _PREPROCESS, _DEVICE


def encode_image_feature(image, model_name: str = "ViT-B/32") -> np.ndarray:
    """
    提取单张图片的 CLIP 特征向量（L2 归一化后的 float32 numpy 数组，shape=(512,)）。
    供失物登记入库复用：feature.tobytes() 存 SQLite BLOB，np.frombuffer(blob) 还原。
    :param image: 图片路径(str/Path) 或 PIL.Image
    """
    model, preprocess, device = load_clip_model(model_name)
    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
    elif isinstance(image, (str, os.PathLike)):
        pil = Image.open(image).convert("RGB")
    else:
        raise TypeError("image 需为图片路径或 PIL.Image")
    tensor = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype(np.float32)[0]


# ================= 核心匹配类 =================
class CLIPMatcher:
    def __init__(self, model_name: str = "ViT-B/32", threshold: float = 0.25):
        """
        初始化 CLIP 匹配器
        :param model_name: CLIP 模型名称，默认 ViT-B/32
        :param threshold: 相似度判定阈值，0~1 区间（建议 0.20~0.30）
        """
        self.model, self.preprocess, self.device = load_clip_model(model_name)
        self.threshold = float(threshold)

    def match_images(self, text_query: str, candidate_images: List[Image.Image],
                     top_k: int = 5) -> dict:
        """
        将文本描述与 YOLO 裁剪出的候选图像列表做语义匹配。
        :param text_query: 用户输入的物品描述，如 "a black backpack"
        :param candidate_images: 候选子图列表 (list of PIL.Image)
        :param top_k: 返回得分最高的前 K 个结果
        :return: dict，含 status / is_found / best_score / top_k_results / message
                 其中 best_score 与 top_k_results 的得分均为 0~1 相似度
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

        top_k = min(max(int(top_k), 1), len(candidate_images))
        text_query = str(text_query).strip()

        try:
            with torch.no_grad():
                # 1. 文本特征（L2 归一化）
                text_input = clip.tokenize([text_query]).to(self.device)
                text_features = self.model.encode_text(text_input)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # 2. 图像特征批量提取（Batch Inference）
                image_inputs = torch.stack([
                    self.preprocess(img.convert("RGB") if isinstance(img, Image.Image) else img)
                    for img in candidate_images
                ]).to(self.device)
                image_features = self.model.encode_image(image_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # 3. 余弦相似度（归一化后点积即为余弦，范围 -1~1，匹配典型 0.2~0.4）
                #    关键点：不再乘 100，保持 0~1 口径，与前端百分比显示和阈值一致
                scores = (text_features @ image_features.T).squeeze(0)
                scores = scores.clamp(0.0, 1.0).cpu().numpy().tolist()

            # 4. 排序与聚合
            results = list(zip(candidate_images, scores))
            results.sort(key=lambda x: x[1], reverse=True)
            top_results = results[:top_k]
            best_score = top_results[0][1] if top_results else 0.0

            # 5. 阈值判定
            is_found = best_score >= self.threshold

            return {
                "status": "success",
                "is_found": is_found,
                "best_score": round(best_score, 4),
                "top_k_results": top_results,   # [(PIL.Image, score)]，score 为 0~1
                "message": "✅ 找到目标！" if is_found else "❌ 未找到目标（相似度未达到阈值）"
            }

        except Exception as e:
            return {
                "status": "error",
                "is_found": False,
                "message": f"图像编码或匹配失败: {str(e)}",
                "best_score": 0.0,
                "top_k_results": []
            }


# ================= 本地独立测试 =================
if __name__ == "__main__":
    img_blue = Image.new('RGB', (224, 224), color='blue')
    img_red = Image.new('RGB', (224, 224), color='red')
    mock_yolo_crops = [img_blue, img_red]

    matcher = CLIPMatcher(threshold=0.20)

    print("\n--- 第一次查询：找蓝色图 ---")
    result = matcher.match_images("a blue picture", mock_yolo_crops, top_k=2)
    print(f"判定结果: {result['message']}")
    print(f"最高得分: {result['best_score']:.4f}  (0~1 余弦相似度)")
    print(f"显示百分比: {result['best_score'] * 100:.1f}%")
    for idx, (img, score) in enumerate(result['top_k_results']):
        print(f"  Top {idx + 1}: 相似度 {score:.4f}")

    print("\n--- 第二次查询：找红色图 (验证单例加载) ---")
    result2 = matcher.match_images("a red picture", mock_yolo_crops, top_k=2)
    print(f"判定结果: {result2['message']}")
    print(f"最高得分: {result2['best_score']:.4f}")

    print("\n--- 第三次：异常输入 ---")
    print(matcher.match_images("", mock_yolo_crops)["message"])
    print(matcher.match_images("a cat", [])["message"])
