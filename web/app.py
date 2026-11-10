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
把测试图片放进 web/test_images/ 文件夹即可，网页会自动扫描。

【LOGO】
网页自动读取 web/logo.png（自动裁白边、转 base64 内嵌），换 LOGO 只替换这张图。
"""

import os
import io
import glob
import time
import base64
import random
import streamlit as st
from PIL import Image, ImageChops

# ============================================================================
# ★★★ 唯一需要修改的开关区 ★★★
# ============================================================================
MOCK_MODE = True  # True = 演示模式(假数据)；YOLO/CLIP 就绪后改为 False

# 联调时取消下面两行注释即可（接口与两位同学的现有代码完全一致）：
# from YOLO import YOLODetector      # 秦宇霄：YOLODetector(model_path, conf_threshold).detect_and_crop(path) -> [PIL子图]
# from CLIP import CLIPMatcher       # 黄涵：CLIPMatcher(threshold).match_images(文本, [PIL子图], top_k) -> dict

IMAGE_DIR = os.path.join("web", "test_images")   # 图库文件夹
LOGO_PATH = os.path.join("web", "logo.png")      # LOGO 文件

st.set_page_config(page_title="慧眼寻物 SmartFind", page_icon="🔍", layout="centered")


# ============================================================================
# ① 样式层：把 Streamlit 默认长相换成设计稿的视觉（纯 CSS，不碰任何逻辑）
# ============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');

/* ---------- 全局：浅蓝渐变 + 柔和光斑 ---------- */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background: linear-gradient(155deg, #f4faff 0%, #e6f3ff 28%, #cfe8ff 58%, #a9d4fa 100%) !important;
    font-family: "Noto Sans SC", "HarmonyOS Sans SC", "PingFang SC",
                 "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
}
.stApp::before, .stApp::after {
    content: ""; position: fixed; border-radius: 50%;
    pointer-events: none; z-index: 0;
}
.stApp::before {
    width: 680px; height: 680px; left: -180px; top: -260px;
    background: radial-gradient(circle at 35% 35%, rgba(255,255,255,.95), rgba(255,255,255,0) 62%);
    filter: blur(8px);
}
.stApp::after {
    width: 720px; height: 720px; right: -220px; bottom: -280px;
    background: radial-gradient(circle, rgba(255,255,255,.7), rgba(255,255,255,0) 66%);
}

/* ---------- 去掉 Streamlit 原生外壳 ---------- */
header[data-testid="stHeader"], #MainMenu, footer,
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }
[data-testid="stAppViewContainer"] > .main { background: transparent; }
[data-testid="InputInstructions"] { display: none !important; }  /* 英文提示条 */
.stApp { --primary-color: #2f74d0 !important; }                  /* 主题色：默认红 -> 品牌蓝 */

/* ---------- 中央白色大卡片 ---------- */
[data-testid="stMainBlockContainer"], .block-container {
    max-width: 660px !important;
    padding: 26px 30px 34px !important;
    margin: 118px auto 70px !important;
    background: #ffffff;
    border-radius: 26px;
    box-shadow: 0 26px 64px rgba(31, 86, 153, .18), 0 2px 10px rgba(31, 86, 153, .06);
    position: relative; z-index: 2;
}
[data-testid="stVerticalBlock"] { gap: .55rem; }

/* ---------- 左上 LOGO 徽标（微微倾斜，像贴纸） ---------- */
.brand-wrap {
    position: fixed; left: 40px; top: 24px; z-index: 9;
    padding: 9px 16px;
    background: rgba(255,255,255,.96);
    border: 1.5px solid #d3e6fb;
    border-radius: 18px;
    box-shadow: 0 10px 26px rgba(31,86,153,.13);
    transform: rotate(-3deg);
}
.brand-wrap img { height: 68px; display: block; }
.brand-wrap .fallback { font-size: 20px; font-weight: 700; color: #1d63b8; letter-spacing: 1px; }

/* ---------- 右上头像 ---------- */
.avatar-wrap {
    position: fixed; right: 46px; top: 32px; z-index: 9;
    width: 54px; height: 54px; border-radius: 50%;
    background: linear-gradient(180deg, #ffffff, #eaf4ff);
    border: 2px solid #d3e6fb;
    box-shadow: 0 8px 20px rgba(31,86,153,.14);
    display: flex; align-items: center; justify-content: center;
}

/* ---------- 搜索胶囊（外框画在容器上，input 本身透明） ---------- */
[data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; }
.st-key-search_form [data-testid="stTextInput"] .react-aria-TextField > div,
.st-key-search_form [data-testid="stTextInput"] div[data-baseweb="input"] {
    position: relative !important;
    height: 58px !important;
    display: flex !important; align-items: center !important;
    border-radius: 30px !important;
    border: 1.6px solid #7aa9dd !important;
    background-color: #ffffff !important;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%233f7bc4' stroke-width='2' stroke-linecap='round'><circle cx='11' cy='11' r='7'/><line x1='16.5' y1='16.5' x2='21' y2='21'/></svg>");
    background-repeat: no-repeat;
    background-position: 22px center;
    background-size: 19px 19px;
    box-shadow: 0 0 0 5px rgba(255,255,255,.75) !important;
    transition: border-color .2s, box-shadow .2s;
}
.st-key-search_form [data-testid="stTextInput"] .react-aria-TextField > div:focus-within,
.st-key-search_form [data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
    border-color: #2f74d0 !important;
    box-shadow: 0 0 0 6px rgba(47,116,208,.12) !important;
}
.st-key-search_form input {
    height: 100% !important;
    border: none !important; outline: none !important;
    background: transparent !important; box-shadow: none !important;
    padding: 0 24px 0 54px !important;
    font-size: 15px !important; color: #24405e !important;
}
.st-key-search_form input::placeholder { color: #93a9c2; }
/* 回车即可搜索，默认提交按钮藏起来 */
.st-key-search_form [data-testid="stFormSubmitButton"] {
    position: absolute !important; left: -9999px; width: 1px; height: 1px; opacity: 0;
}
[data-testid="stTextInput"] label { display: none; }

/* ---------- 分区小标题 ---------- */
.sec {
    display: flex; justify-content: space-between; align-items: center;
    margin: 20px 0 6px; font-size: 14px; font-weight: 600; color: #4a6d94;
    letter-spacing: .5px;
}

/* ---------- 历史记录（隔行浅蓝） ---------- */
.hist { display: flex; flex-direction: column; gap: 2px; }
.hrow {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 18px; font-size: 14.5px; color: #2c405c; border-radius: 8px;
}
.hrow.odd { background: #f0f8ff; }
.hrow .q { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 74%; }
.hrow .s { font-size: 13.5px; font-weight: 500; white-space: nowrap; }
.s-ok { color: #17a05e; }
.s-run { color: #2f74d0; }
.s-no { color: #98a6b8; }

/* “折叠 / 展开”文字按钮 */
.st-key-hist_toggle button {
    background: transparent !important; border: none !important;
    color: #6b8bad !important; font-size: 13px !important; font-weight: 400 !important;
    padding: 0 !important; min-height: 20px !important; height: 20px !important;
    line-height: 20px !important; box-shadow: none !important;
}
.st-key-hist_toggle button:hover { color: #2f74d0 !important; }

/* ---------- 备选图片九宫格 ---------- */
.grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 6px; }
.tile {
    position: relative; border-radius: 13px; overflow: hidden;
    aspect-ratio: 4 / 3; background: #e8e8e8;
    display: flex; align-items: center; justify-content: center;
    color: #b0b0b0; font-size: 12.5px; letter-spacing: .5px;
}
.tile img { width: 100%; height: 100%; object-fit: cover; display: block; transition: transform .4s ease; }
.tile:hover img { transform: scale(1.09); }
.tile .chip {
    position: absolute; top: 6px; right: 6px; z-index: 2;
    background: rgba(255,255,255,.92); color: #2f6fc4;
    font-size: 10.5px; font-weight: 700; padding: 2px 7px; border-radius: 999px;
}
.tile .fn {
    position: absolute; left: 0; right: 0; bottom: 0; z-index: 2;
    padding: 16px 8px 6px; font-size: 10.5px; color: #fff;
    background: linear-gradient(transparent, rgba(0,0,0,.58));
    opacity: 0; transition: opacity .25s;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.tile:hover .fn { opacity: 1; }

/* ---------- 结果小结 ---------- */
.sum { font-size: 12.5px; color: #8ea4bd; margin-top: 10px; text-align: right; }
.sum b { color: #2f74d0; font-weight: 700; }

/* ---------- 右下角悬浮设置按钮 ---------- */
.st-key-settings [data-testid="stPopover"] button {
    position: fixed; right: 26px; bottom: 26px; z-index: 20;
    width: 48px; height: 48px; border-radius: 50%;
    background: #fff !important; border: 1.5px solid #d7e6f7 !important;
    color: #4a7cb8 !important; font-size: 17px !important;
    box-shadow: 0 8px 22px rgba(31,86,153,.16);
    padding: 0 !important; min-height: 48px;
}
.st-key-settings [data-testid="stPopover"] button:hover {
    border-color: #7aa9dd !important; color: #2f74d0 !important;
}
[data-testid="stPopoverBody"] { border-radius: 16px; box-shadow: 0 18px 40px rgba(31,86,153,.18); }

/* 弹层里的滑块跟着品牌蓝走 */
[data-testid="stSlider"] div:has(> [data-testid="stSliderThumbValue"]) > div:not([data-testid]) {
    background: #2f74d0 !important;
}
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { color: #2f74d0 !important; }

[data-testid="stAlert"] { border-radius: 12px; font-size: 13.5px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================================
# ② 资源层：图片 -> base64（自定义 HTML 才能直接引用本地图片）
# ============================================================================
@st.cache_data(show_spinner=False)
def _logo_b64(path: str = LOGO_PATH, max_w: int = 560):
    """读 LOGO，自动裁掉四周白边并缩放。"""
    if not os.path.exists(path):
        return None
    try:
        im = Image.open(path).convert("RGB")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        diff = ImageChops.difference(im, bg).convert("L").point(lambda v: 255 if v > 14 else 0)
        bbox = diff.getbbox()
        if bbox:
            im = im.crop(bbox)
        if im.width > max_w:
            im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _thumb_b64(path: str, mtime: float, size: int = 420):
    """结果图裁方图 + 压缩，避免大图拖慢页面。"""
    try:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        im = im.resize((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _grid_html(items, slots: int = 8) -> str:
    """九宫格：有结果出图，没结果出“备选图片”占位块。"""
    tiles = []
    for r in (items or []):
        src = _thumb_b64(r["source"], os.path.getmtime(r["source"]))
        chip = f'<span class="chip">{r["score"] * 100:.0f}%</span>' if r.get("score") else ""
        if src:
            tiles.append(
                f'<div class="tile"><img src="data:image/jpeg;base64,{src}" alt="">'
                f'{chip}<span class="fn">{os.path.basename(r["source"])}</span></div>'
            )
        else:
            tiles.append('<div class="tile">备选图片</div>')
    tiles += ['<div class="tile">备选图片</div>'] * max(0, slots - len(tiles))
    return f'<div class="grid">{"".join(tiles)}</div>'


# ============================================================================
# ③ 数据层：图库扫描 / 模拟检索 / 正式检索（逻辑与上一版完全一致）
# ============================================================================
def scan_library():
    paths = []
    if os.path.isdir(IMAGE_DIR):
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    return sorted(paths)


def mock_search(text, paths, k):
    """演示模式：文本+文件名 做哈希种子，结果稳定可复现，方便录屏演示。"""
    results = []
    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        w, h = img.size
        seed = int.from_bytes((text + os.path.basename(path)).encode("utf-8")[:8], "little")
        rng = random.Random(seed ^ int("smartfind", 36))
        bw, bh = rng.randint(w // 5, w // 3), rng.randint(h // 5, h // 3)
        x1 = rng.randint(w // 10, max(w // 10 + 1, w - bw - w // 10))
        y1 = rng.randint(h // 10, max(h // 10 + 1, h - bh - h // 10))
        box = (x1, y1, x1 + bw, y1 + bh)
        results.append({
            "source": path, "orig": img, "crop": img.crop(box), "box": box,
            "score": round(rng.uniform(0.35, 0.95), 4),
            "yolo_conf": round(rng.uniform(0.55, 0.95), 2),
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]


@st.cache_resource
def load_models():
    """模型只加载一次，多次查询秒回。"""
    detector = YOLODetector(model_path="models/yolov8n.pt", conf_threshold=0.15)
    matcher = CLIPMatcher(threshold=0.25)
    return detector, matcher


def real_search(text, paths, k, thresh):
    """正式流程：YOLO 切图 -> CLIP 打分 -> 排序。联调时只需保证这里能跑通。"""
    detector, matcher = load_models()
    matcher.threshold = thresh
    results = []
    for path in paths:
        crops = detector.detect_and_crop(path)
        if not crops:
            continue
        res = matcher.match_images(text, crops, top_k=k)
        if res.get("status") != "success":
            continue
        try:
            orig = Image.open(path).convert("RGB")
        except Exception:
            orig = None
        for crop, score in res["top_k_results"]:
            results.append({
                "source": path, "orig": orig, "crop": crop, "box": None,
                "score": round(float(score), 4), "yolo_conf": None,
            })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]


# ============================================================================
# ④ 交互层：页面装配
# ============================================================================
ss = st.session_state
ss.setdefault("history", [])      # [{"q": 文本, "status": 查询中/已找到/未找到}]
ss.setdefault("result", None)     # {"q": 文本, "items": [...], "best": 分数}
ss.setdefault("pending", None)    # 待执行的查询
ss.setdefault("hist_open", True)

logo = _logo_b64()
library = scan_library()

# 左上 LOGO
if logo:
    st.markdown(f'<div class="brand-wrap"><img src="data:image/png;base64,{logo}" alt="慧眼寻物 SmartFind"></div>',
                unsafe_allow_html=True)
else:
    st.markdown('<div class="brand-wrap"><span class="fallback">慧眼寻物 SmartFind</span></div>',
                unsafe_allow_html=True)

# 右上头像
st.markdown("""
<div class="avatar-wrap">
  <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#8b98a8" stroke-width="1.6">
    <circle cx="12" cy="8.2" r="3.6"/><path d="M4.6 20c1.2-3.9 4-5.8 7.4-5.8s6.2 1.9 7.4 5.8"/>
  </svg>
</div>
""", unsafe_allow_html=True)

# 搜索框（回车提交）
with st.container(key="search_form"):
    with st.form("search", clear_on_submit=False):
        query = st.text_input("q", value="", placeholder="输入名称、特征，搜索失物",
                              label_visibility="collapsed")
        submitted = st.form_submit_button("搜索")

if submitted and query.strip():
    ss.history.insert(0, {"q": query.strip(), "status": "查询中"})
    ss.history = ss.history[:6]
    ss.pending = query.strip()
    st.rerun()

# 待执行查询：先让“查询中”露个脸，再跑检索并翻成最终状态
if ss.pending:
    pending = ss.pending
    ss.pending = None
    time.sleep(0.6)  # 联调时可删掉这行
    with st.spinner("正在检索图库..."):
        items = (mock_search(pending, library, ss.get("top_k", 3)) if MOCK_MODE
                 else real_search(pending, library, ss.get("top_k", 3), ss.get("threshold", 0.25)))
    best = items[0]["score"] if items else 0.0
    ss.result = {"q": pending, "items": items, "best": best}
    for h in ss.history:
        if h["q"] == pending:
            h["status"] = "已找到" if best >= ss.get("threshold", 0.25) else "未找到"
            break
    st.rerun()

# 历史记录
head_l, head_r = st.columns([4, 1], vertical_alignment="center")
with head_l:
    st.markdown('<div class="sec" style="margin:20px 0 4px">历史记录</div>', unsafe_allow_html=True)
with head_r:
    with st.container(key="hist_toggle"):
        if st.button("折叠" if ss.hist_open else "展开", use_container_width=True):
            ss.hist_open = not ss.hist_open
            st.rerun()

if ss.hist_open:
    if ss.history:
        rows = []
        for i, h in enumerate(ss.history):
            cls = {"已找到": "s-ok", "查询中": "s-run"}.get(h["status"], "s-no")
            rows.append(
                f'<div class="hrow{" odd" if i % 2 == 0 else ""}">'
                f'<span class="q">{h["q"]}</span>'
                f'<span class="s {cls}">{h["status"]}</span></div>'
            )
        st.markdown(f'<div class="hist">{"".join(rows)}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="hist"><div class="hrow odd">'
                    '<span class="q" style="color:#a8b6c6">暂无搜索记录</span>'
                    '<span class="s s-no">—</span></div></div>', unsafe_allow_html=True)

# 备选图片
st.markdown('<div class="sec" style="margin:22px 0 4px">备选图片</div>', unsafe_allow_html=True)
res = ss.result
st.markdown(_grid_html(res["items"] if res else None), unsafe_allow_html=True)

if res and res["items"]:
    st.markdown(f'<div class="sum">「{res["q"]}」共 {len(res["items"])} 张备选画面 · '
                f'最高相似度 <b>{res["best"] * 100:.1f}%</b></div>', unsafe_allow_html=True)
elif res:
    st.markdown(f'<div class="sum">「{res["q"]}」未在 {len(library)} 张图库中找到匹配画面</div>',
                unsafe_allow_html=True)

# 右下角设置
with st.container(key="settings"):
    with st.popover("⚙", help="检索参数"):
        st.caption(f"图库：{len(library)} 张 · {'演示模式' if MOCK_MODE else 'YOLO + CLIP 已接入'}")
        ss["top_k"] = st.slider("返回结果数量", 1, 12, ss.get("top_k", 3))
        ss["threshold"] = st.slider("相似度阈值", 0.05, 0.60, ss.get("threshold", 0.25), 0.01)
        st.caption(f"图库目录：`{IMAGE_DIR}`")
