"""
慧眼寻物 SmartFind —— 网页前端（网页开发负责人维护）
====================================================
运行方式：在项目根目录 D:/SmartFind 下执行
    streamlit run web/app.py

【现在 -> 将来 只需两步切换】
1. 测试阶段（现在）：MOCK_MODE = True，用模拟数据展示完整效果；
2. 联调阶段（将来）：MOCK_MODE = False，并取消下方两行 import 的注释，
   即接入 YOLO 同学（秦宇霄）的 YOLODetector 和 CLIP 同学（黄涵）的 CLIPMatcher。
   其余所有界面代码完全不用改。

【数据集就位后】
把测试图片放进 web/test_images/ 文件夹即可，网页会自动扫描，
不需要修改任何代码。
"""

import os
import glob
import random
import streamlit as st
from PIL import Image, ImageDraw

# ============================================================================
# ★★★ 唯一需要修改的开关区 ★★★
# ============================================================================
MOCK_MODE = True  # True = 演示模式(假数据)；YOLO/CLIP 就绪后改为 False

# 联调时取消下面两行注释即可（接口与两位同学的现有代码完全一致）：
# from YOLO import YOLODetector      # 秦宇霄：YOLODetector(model_path, conf_threshold).detect_and_crop(path) -> [PIL子图]
# from CLIP import CLIPMatcher       # 黄涵：CLIPMatcher(threshold).match_images(文本, [PIL子图], top_k) -> dict

# 测试图片文件夹（数据集搜集完毕后，把图片丢进这个文件夹即可自动生效）
IMAGE_DIR = os.path.join("web", "test_images")

# ============================================================================
# 以下全部是界面与流水线代码，联调时无需改动
# ============================================================================

st.set_page_config(page_title="慧眼寻物 SmartFind", page_icon="🔍", layout="wide")

# ---------------- 侧边栏：系统设置 ----------------
with st.sidebar:
    st.header("⚙️ 系统设置")
    top_k = st.slider("返回最匹配的结果数量", 1, 10, 3)
    threshold = st.slider("CLIP 相似度判定阈值", 0.05, 0.60, 0.25, 0.01,
                          help="得分超过该阈值才判定为“找到目标”，建议 0.2~0.3")
    st.markdown("---")

    # 自动扫描图片文件夹
    img_paths = []
    if os.path.isdir(IMAGE_DIR):
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            img_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
        img_paths.sort()

    if img_paths:
        st.success(f"✅ 已加载图库：{len(img_paths)} 张图片")
        st.caption(f"目录：`{IMAGE_DIR}`\n\n数据集就位后把图片放入该文件夹即可。")
        with st.expander("查看图库列表"):
            for p in img_paths:
                st.caption(os.path.basename(p))
    else:
        st.error(f"图库为空：未在 `{IMAGE_DIR}` 找到任何图片，请先放入测试图片。")

    st.markdown("---")
    st.info("🧪 当前模式：**演示模式（模拟数据）**" if MOCK_MODE
            else "🚀 当前模式：**正式联调（YOLO + CLIP 已接入）**")

st.title("🔍 慧眼寻物 SmartFind 智能失物检索系统")
st.write("输入您丢失的物品描述，系统将从监控图库中为您检索最匹配的画面。")

# ---------------- 顶部输入区 ----------------
col_input, col_btn = st.columns([4, 1])
with col_input:
    query = st.text_input("请输入丢失物品描述",
                          placeholder="例如：黑色背包、蓝色雨伞、一台笔记本电脑...")
with col_btn:
    st.write("")
    search_clicked = st.button("开始查找 🔍", use_container_width=True, type="primary")

# ============================================================================
# 模拟检索（演示模式）：根据 query+图片名 生成稳定可复现的假结果
# 说明：用哈希做种子，同样的描述搜同样的图，结果永远一致，方便录屏演示
# ============================================================================
def mock_search(text, paths, k):
    results = []
    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        w, h = img.size

        # 用 md5 哈希做伪随机种子：结果稳定，不同查询/图片之间又有区分度
        seed = int.from_bytes(
            (text + os.path.basename(path)).encode("utf-8")[:8], "little")
        rng = random.Random(seed ^ int(text[:0] + "smartfind", 36))

        # 生成一个画面中部的模拟检测框（保证不出界、不会小到看不见）
        bw = rng.randint(w // 5, w // 3)
        bh = rng.randint(h // 5, h // 3)
        x1 = rng.randint(w // 10, max(w // 10 + 1, w - bw - w // 10))
        y1 = rng.randint(h // 10, max(h // 10 + 1, h - bh - h // 10))
        box = (x1, y1, x1 + bw, y1 + bh)

        score = round(rng.uniform(0.35, 0.95), 4)
        crop = img.crop(box)

        results.append({
            "source": path, "orig": img, "crop": crop,
            "box": box, "score": score,
            "yolo_conf": round(rng.uniform(0.55, 0.95), 2),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]

# ============================================================================
# 正式检索（联调模式）：YOLO 切图 -> CLIP 打分 -> 排序
# 以后要改的只有这一个函数，界面层永远不用动
# ============================================================================
@st.cache_resource
def load_models():
    """模型加载做了缓存：整个网页生命周期只加载一次，多次查询秒回。"""
    detector = YOLODetector(model_path="models/yolov8n.pt", conf_threshold=0.15)
    matcher = CLIPMatcher(threshold=0.25)
    return detector, matcher

def real_search(text, paths, k, thresh):
    detector, matcher = load_models()
    matcher.threshold = thresh  # 同步侧边栏阈值
    results = []

    progress = st.progress(0.0, text="正在检索图库...")
    for i, path in enumerate(paths):
        # ① YOLO 阶段：切出候选子图（PIL 列表）
        crops = detector.detect_and_crop(path)

        # ② CLIP 阶段：文本与子图批量匹配
        if crops:
            res = matcher.match_images(text, crops, top_k=k)
            if res["status"] == "success":
                try:
                    orig = Image.open(path).convert("RGB")
                except Exception:
                    orig = None
                for crop, score in res["top_k_results"]:
                    # 说明：目前 YOLO 同学的 detect_and_crop 只返回子图、不返回框坐标，
                    #      所以联调阶段先直接展示子图；后续若接口加上 (crop, box)，
                    #      把下面 box 换成真实坐标即可自动在原图上画红框。
                    results.append({
                        "source": path, "orig": orig, "crop": crop,
                        "box": None, "score": round(float(score), 4),
                        "yolo_conf": None,
                    })
        progress.progress((i + 1) / len(paths),
                          text=f"正在检索图库... ({i + 1}/{len(paths)})")
    progress.empty()

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]

# ---------------- 搜索逻辑 ----------------
if search_clicked:
    if not query.strip():
        st.warning("⚠️ 请先输入物品描述再点击查找！")
    elif not img_paths:
        st.error("图库为空，请先往 web/test_images/ 放入测试图片。")
    else:
        with st.spinner("检索中，请稍候..."):
            if MOCK_MODE:
                results = mock_search(query.strip(), img_paths, top_k)
            else:
                results = real_search(query.strip(), img_paths, top_k, threshold)

        st.markdown("---")
        best = results[0]["score"] if results else 0.0
        if best >= threshold:
            st.success(f"🎯 检索完成：**很可能找到了「{query.strip()}」**，"
                       f"最高相似度 {best:.2%}")
        else:
            st.warning(f"❓ 检索完成：图库中暂未发现与「{query.strip()}」高度匹配的物体，"
                       f"最高相似度仅 {best:.2%}（阈值 {threshold:.0%}）")

        st.subheader(f"Top {len(results)} 检索结果")

        # 三列网格展示
        cols = st.columns(3)
        for idx, r in enumerate(results):
            with cols[idx % 3]:
                with st.container(border=True):
                    # 联调模式若无框坐标则直接展示 YOLO 切出的子图
                    if r["box"] is not None and r["orig"] is not None:
                        shown = r["orig"].copy()
                        ImageDraw.Draw(shown).rectangle(r["box"],
                                                        outline="red", width=6)
                        st.image(shown, use_container_width=True)
                    else:
                        st.image(r["crop"], use_container_width=True)

                    st.markdown(f"**结果 {idx + 1}** · `{os.path.basename(r['source'])}`")
                    st.progress(min(r["score"], 1.0),
                                text=f"CLIP 相似度：{r['score']:.2%}")
                    if r["yolo_conf"] is not None:
                        st.caption(f"YOLO 置信度：{r['yolo_conf']:.2f}")

                    with st.expander("查看候选子图"):
                        st.image(r["crop"], use_container_width=True)
else:
    st.markdown("---")
    st.info("👈 在上方输入物品描述并点击「开始查找」。"
            "演示模式下任意描述都会返回模拟结果，用于预览最终展示效果。")
