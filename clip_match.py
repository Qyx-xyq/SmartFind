"""
clip_match.py —— M5 图文语义匹配模块（CLIP 匹配流程）

职责：输入一段物品文字描述 + YOLO 裁剪出的候选子图列表，
      用 CLIP 计算图文余弦相似度，返回按相似度降序排列的 Top-K 结果，
      并按阈值判定图中是否存在目标物品。

说明：本模块只负责 CLIP 匹配，不涉及 YOLO 检测，也不涉及网页界面，
     供流水线 / 界面层直接调用。
"""

import torch
import clip
from PIL import Image

# 全局模型单例，整个进程只加载一次，避免重复加载拖慢速度
_clip_model = None
_preprocess = None
_device = None


def load_clip_model(model_name="ViT-B/32"):
    """加载 CLIP 模型（只加载一次），返回 (模型, 预处理, 设备)。"""
    global _clip_model, _preprocess, _device
    if _clip_model is not None:
        return _clip_model, _preprocess, _device

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print("正在加载 CLIP 模型...")
    _clip_model, _preprocess = clip.load(model_name, device=_device)
    return _clip_model, _preprocess, _device


def _encode_text(text: str, model, device):
    """把文字描述编码为归一化文本特征向量。"""
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        feat = model.encode_text(tokens)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


def _encode_image(pil_img: Image.Image, model, preprocess, device):
    """把单张候选子图编码为归一化图像特征向量。"""
    inp = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(inp)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


def clip_match(text, crops, top_k=3, threshold=0.25, model=None, preprocess=None, device=None):
    """
    CLIP 图文匹配主函数。

    参数:
        text:      用户输入的物品文字描述（如 "a photo of a bus"）
        crops:     YOLO 裁剪出的 PIL 候选子图列表
        top_k:     返回相似度最高的前 K 个结果
        threshold: 判定阈值，最高分 >= 阈值即视为找到目标
        model / preprocess / device:
                   可选，传入流水线已加载好的 CLIP 模型以复用，避免重复加载；
                   不传则模块内部自行加载。

    返回:
        dict: {
            "found":     bool,            # 是否判定找到目标
            "best_score": float,          # 最高相似度得分
            "results":   list,            # 按相似度降序的 Top-K：[{"rank","score","crop"}]
            "message":   str,            # 判定结果 / 异常提示
        }
    """
    # —— 异常处理 1：输入文本为空 ——
    if text is None or not str(text).strip():
        return {
            "found": False,
            "best_score": 0.0,
            "results": [],
            "message": "输入文本为空，请重新输入物品描述",
        }

    # —— 异常处理 2：没有收到 YOLO 候选子图 ——
    if not crops:
        return {
            "found": False,
            "best_score": 0.0,
            "results": [],
            "message": "未收到候选子图（YOLO 未检测到任何物体）",
        }

    # 复用外部已加载模型，否则模块内部自行加载
    if model is None:
        model, preprocess, device = load_clip_model()

    text = str(text).strip()
    text_feat = _encode_text(text, model, device)

    # 逐张子图编码并计算相似度，单张失败则跳过，不影响其余
    scored = []
    for crop in crops:
        try:
            img_feat = _encode_image(crop, model, preprocess, device)
            sim = float((img_feat @ text_feat.T).item())
            scored.append({"crop": crop, "score": sim})
        except Exception:
            continue

    if not scored:
        return {
            "found": False,
            "best_score": 0.0,
            "results": [],
            "message": "候选子图编码失败，无法完成匹配",
        }

    # —— 按相似度从高到低排序，取 Top-K ——
    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:top_k]
    best_score = top[0]["score"]
    found = best_score >= threshold

    results = [
        {"rank": i + 1, "score": round(item["score"], 4), "crop": item["crop"]}
        for i, item in enumerate(top)
    ]

    message = "✅ 找到目标！" if found else "❌ 未找到目标（相似度未达到阈值）"
    return {
        "found": found,
        "best_score": round(best_score, 4),
        "results": results,
        "message": message,
    }
