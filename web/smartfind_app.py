"""
慧眼寻物 SmartFind —— 正式版图文检索界面（YOLO + CLIP 真实推理，无假数据）
=========================================================================
放置位置：D:/SmartFind/web/smartfind_app.py
"""

from __future__ import annotations

import base64
import glob
import html
import io
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import streamlit as st
from PIL import Image, ImageChops

# ============================================================================
# 路径与配置
# ============================================================================
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if (_HERE.parent / "YOLO.py").exists() else _HERE

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

LIBRARY_DIRS = [
    # ROOT / "data",                       # data/1.jpg 2.jpg 3.jpg bottle+laptop.jpg bus.jpg
    ROOT / "web" / "test_images",      # 想扩图库时把这一行放开即可
]
ADMIN_UPLOAD_DIR = ROOT / "web" / "admin_uploads"

YOLO_WEIGHTS = ROOT / "models" / "yolov8n.pt"
YOLO_CONF = 0.15
DEFAULT_THRESHOLD = 0.25
DEFAULT_TOPK = 8
GRID_SLOTS = 8
LOGO_PATHS = [ROOT / "web" / "logo.png", ROOT / "logo.png", _HERE / "logo.png"]

# ============================================================================
# ★ 数据源适配层
# ============================================================================
@dataclass
class Candidate:
    key: str
    image: Image.Image
    label: str
    source: str = "file"
    path: Optional[Path] = None
    meta: dict = field(default_factory=dict)


def scan_images(directory: Path) -> List[Path]:
    if not directory or not directory.is_dir():
        return []
    found: List[Path] = []
    for ext in IMAGE_EXTS:
        found.extend(glob.glob(str(directory / f"*{ext}")))
        found.extend(glob.glob(str(directory / f"*{ext.upper()}")))
    out, seen = [], set()
    for p in sorted(found):
        if p in seen: continue
        seen.add(p)
        try:
            if os.path.getsize(p) > 0:
                out.append(Path(p))
        except OSError:
            continue
    return out


def candidate_from_path(path: Path, label: Optional[str] = None, source: str = "file") -> Optional[Candidate]:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    return Candidate(
        key=str(path),
        image=img,
        label=label or Path(path).stem,
        source=source,
        path=Path(path),
        meta={"mtime": os.path.getmtime(path), "size": os.path.getsize(path)},
    )


def load_candidates_from_dir(directory: Path, source: str = "admin_upload") -> List[Candidate]:
    items = []
    for p in scan_images(directory):
        c = candidate_from_path(p, source=source)
        if c: items.append(c)
    return items


def load_candidates() -> List[Candidate]:
    items: List[Candidate] = []
    for d in LIBRARY_DIRS:
        items.extend(load_candidates_from_dir(d, source="file"))
    if ADMIN_UPLOAD_DIR.is_dir():
        items.extend(load_candidates_from_dir(ADMIN_UPLOAD_DIR, source="admin_upload"))
    return items

# ============================================================================
# 算法层：YOLO 裁片 -> CLIP 打分
# ============================================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    from YOLO import YOLODetector
    from CLIP import CLIPMatcher
    detector = YOLODetector(model_path=str(YOLO_WEIGHTS), conf_threshold=YOLO_CONF)
    matcher = CLIPMatcher(threshold=DEFAULT_THRESHOLD)
    return detector, matcher

_CROP_CACHE: dict = {}

def crops_for(detector, cand: Candidate) -> List[Image.Image]:
    ck = (cand.key, cand.meta.get("mtime"), cand.meta.get("size"))
    if ck in _CROP_CACHE:
        return _CROP_CACHE[ck]
    target = str(cand.path) if cand.path else cand.image
    crops = detector.detect_and_crop(target) or []
    _CROP_CACHE[ck] = crops
    return crops


def run_search(query: str, candidates: List[Candidate], *, use_yolo: bool = True,
               top_k: int = DEFAULT_TOPK, threshold: float = DEFAULT_THRESHOLD) -> dict:
    detector, matcher = get_engine()
    matcher.threshold = threshold

    pool: List[Image.Image] = []
    owner: List[str] = []
    info: dict = {}

    for cand in candidates:
        try:
            crops = crops_for(detector, cand) if use_yolo else []
        except Exception:
            crops = []
        detected = bool(crops)
        if not crops:
            crops = [cand.image]
        info[cand.key] = {"cand": cand, "n_crops": len(crops), "detected": detected}
        for crop in crops:
            pool.append(crop)
            owner.append(cand.key)

    if not pool:
        return {"q": query, "items": [], "best": 0.0, "n_hit": 0,
                "n_lib": len(candidates), "status": "empty", "msg": "图库为空"}

    res = matcher.match_images(query, pool, top_k=len(pool))
    if res["status"] != "success":
        return {"q": query, "items": [], "best": 0.0, "n_hit": 0,
                "n_lib": len(candidates), "msg": res["message"], "status": "error"}

    idx_by_id = {id(crop): owner[i] for i, crop in enumerate(pool)}
    grouped: dict = {}
    for crop, score in res["top_k_results"]:
        key = idx_by_id.get(id(crop))
        if key is None: continue
        grouped.setdefault(key, []).append((crop, float(score)))

    items = []
    for key, hits in grouped.items():
        hits.sort(key=lambda x: x[1], reverse=True)
        best_crop, best_score = hits[0]
        cand = info[key]["cand"]
        items.append({
            "key": key, "label": cand.label, "source": cand.source,
            "score": best_score, "hit": best_score >= threshold,
            "thumb": best_crop, "full": cand.image,
            "n_crops": info[key]["n_crops"], "detected": info[key]["detected"],
            "debug": hits[:4],
        })

    items.sort(key=lambda r: r["score"], reverse=True)
    best = items[0]["score"] if items else 0.0
    n_hit = sum(1 for r in items if r["hit"])
    return {
        "q": query, "items": items[:top_k], "best": best,
        "n_hit": n_hit, "n_lib": len(candidates),
        "msg": f"命中 {n_hit} 张" if n_hit else "未找到达到阈值的照片",
        "status": "ok",
    }

# ============================================================================
# 展示层：样式 / 图片编码 / HTML 小组件
# ============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');

/* ---------- 背景 ---------- */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background:
        radial-gradient(900px 520px at 6% 4%, #ffffff 0%, rgba(255,255,255,0) 62%),
        radial-gradient(820px 480px at 92% 96%, #ffffff 0%, rgba(255,255,255,0) 58%),
        linear-gradient(158deg, #eef8ff 0%, #dcefff 38%, #bfe1fb 100%) !important;
    font-family: "Noto Sans SC", "HarmonyOS Sans SC", "PingFang SC", sans-serif;
}

/* ---------- 原生外壳隐藏（保留侧边栏以便主控菜单显示） ---------- */
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="InputInstructions"] { display: none !important; }
[data-testid="stAppViewContainer"] > .main { background: transparent; }
.stApp { --primary-color: #2f74d0 !important; }

/* ---------- 中央白色卡片 ---------- */
[data-testid="stMainBlockContainer"], .block-container {
    max-width: 700px !important;
    padding: 24px 28px 30px !important;
    margin: 128px auto 78px !important;
    background: #ffffff;
    border-radius: 26px;
    box-shadow: 0 28px 68px rgba(31, 86, 153, .16), 0 2px 10px rgba(31, 86, 153, .06);
    position: relative; z-index: 2;
}
[data-testid="stVerticalBlock"] { gap: .45rem; }

/* ---------- 左上 LOGO（贴纸效果） ---------- */
.brand {
    position: fixed; left: 38px; top: 22px; z-index: 9;
    padding: 10px 14px; background: rgba(255,255,255,.97);
    border: 1.5px solid #d6e8fb; border-radius: 16px;
    box-shadow: 0 10px 26px rgba(31,86,153,.12);
    transform: rotate(-3deg);
}
.brand img { height: 62px; display: block; }
.brand .txt { font-size: 19px; font-weight: 700; color: #1d63b8; letter-spacing: .5px; }

/* ---------- 右上头像 ---------- */
.avatar {
    position: fixed; right: 42px; top: 30px; z-index: 9;
    width: 54px; height: 54px; border-radius: 50%;
    background: linear-gradient(180deg, #ffffff, #eaf4ff);
    border: 2px solid #d6e8fb; box-shadow: 0 8px 20px rgba(31,86,153,.13);
    display: flex; align-items: center; justify-content: center;
}

/* ---------- 搜索胶囊 ---------- */
[data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; }
.st-key-search [data-testid="stTextInput"] :is([data-baseweb="input"], .react-aria-TextField > div) {
    position: relative !important; height: 56px !important;
    display: flex !important; align-items: center !important;
    border-radius: 30px !important; border: 1.6px solid #8ab4e0 !important;
    background-color: #ffffff !important;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%233f7bc4' stroke-width='1.9' stroke-linecap='round'><circle cx='11' cy='11' r='7'/><line x1='16.4' y1='16.4' x2='21' y2='21'/></svg>");
    background-repeat: no-repeat; background-position: 20px center; background-size: 19px 19px;
    box-shadow: 0 4px 18px rgba(47,116,208,.07) !important;
}
.st-key-search input {
    height: 100% !important; border: none !important;
    padding: 0 22px 0 50px !important; font-size: 15px !important;
}
[data-testid="stTextInput"] label { display: none !important; }

/* ---------- 分区标题与历史记录 ---------- */
.sec { display: flex; justify-content: space-between; margin: 20px 2px 8px; font-size: 13.5px; color: #4a6d94; }
.sec .count { color: #93a9c2; font-size: 12.5px; }
.hist { display: flex; flex-direction: column; gap: 2px; }
.hrow { display: flex; justify-content: space-between; padding: 11px 16px; font-size: 14.5px; border-radius: 8px; }
.hrow.odd { background: #f1f8ff; }

/* ---------- 结果九宫格 ---------- */
.grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 4px; }
.tile { position: relative; border-radius: 13px; overflow: hidden; aspect-ratio: 1 / 1; background: #e8e8e8; display: flex; align-items: center; justify-content: center; }
.tile img { width: 100%; height: 100%; object-fit: cover; }
.tile .chip { position: absolute; top: 6px; right: 6px; font-size: 10.5px; padding: 2px 7px; border-radius: 999px; background: rgba(47,116,208,.94); color: #fff; }
.tile.empty { background: #fbfdff; border: 1.5px dashed #c5d8ea; color: #b9c9d9; }
</style>
"""

@st.cache_data(show_spinner=False)
def _logo_b64(path_str: str, max_w: int = 520) -> Optional[str]:
    if not os.path.exists(path_str): return None
    try:
        im = Image.open(path_str).convert("RGB")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        diff = ImageChops.difference(im, bg).convert("L").point(lambda v: 255 if v > 12 else 0)
        box = diff.getbbox()
        if box: im = im.crop(box)
        if im.width > max_w: im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None

def _b64_image(img: Image.Image, max_side: int = 460, quality: int = 82) -> str:
    im = img.convert("RGB")
    w, h = im.size
    scale = max_side / max(w, h)
    if scale < 1: im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()

def _tile_html(item: dict, show_full: bool = False) -> str:
    src = _b64_image(item["full"] if show_full else item["thumb"])
    pct = round(item["score"] * 100)
    cls = "" if item["hit"] else " ref"
    return (
        f'<div class="tile{cls}"><img src="data:image/jpeg;base64,{src}">'
        f'<span class="chip{cls}">{pct}%</span></div>'
    )

_PLACEHOLDER = '<div class="tile empty">备选图片</div>'

def _grid_html(items: List[dict], slots: int = GRID_SLOTS, show_full: bool = False) -> str:
    tiles = [_tile_html(it, show_full) for it in items[:slots]]
    tiles.extend([_PLACEHOLDER] * max(0, slots - len(tiles)))
    return f'<div class="grid">{"".join(tiles)}</div>'

# ============================================================================
# 页面装配
# ============================================================================
def main() -> None:
    # st.set_page_config(page_title="慧眼寻物 SmartFind", page_icon="🔍", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)

    ss = st.session_state
    ss.setdefault("history", [])
    ss.setdefault("result", None)
    ss.setdefault("pending", None)
    ss.setdefault("hist_open", True)
    ss.setdefault("top_k", DEFAULT_TOPK)
    ss.setdefault("threshold", DEFAULT_THRESHOLD)
    ss.setdefault("use_yolo", True)
    ss.setdefault("show_full", False)
    ss.setdefault("show_debug", False)

    # 顶部 LOGO 和头像
    logo_b64 = None
    for p in LOGO_PATHS:
        logo_b64 = _logo_b64(str(p))
        if logo_b64: break
    if logo_b64:
        st.markdown(f'<div class="brand"><img src="data:image/png;base64,{logo_b64}"></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="brand"><span class="txt">慧眼寻物 SmartFind</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="avatar"><svg width="29" height="29" viewBox="0 0 24 24" fill="none" stroke="#8b98a8" stroke-width="1.6"><circle cx="12" cy="8.2" r="3.6"/><path d="M4.6 20c1.2-3.9 4-5.8 7.4-5.8s6.2 1.9 7.4 5.8"/></svg></div>', unsafe_allow_html=True)

    # 搜索栏
    with st.container(key="search"):
        with st.form("search_form", clear_on_submit=False):
            query = st.text_input("q", value="", label_visibility="collapsed", placeholder="输入名称、特征，搜索失物")
            submitted = st.form_submit_button("搜索")

    if submitted:
        text = (query or "").strip()
        if not text:
            st.warning("请先输入物品名称或特征，例如「红色双肩包」。")
        else:
            ss.history.insert(0, {"q": text, "status": "查询中", "score": None})
            ss.history = ss.history[:8]
            ss.pending = text
            st.rerun()

    candidates = load_candidates()

    # 历史记录
    head_l, head_r = st.columns([4, 1], vertical_alignment="center")
    with head_l:
        st.markdown('<div class="sec" style="margin:18px 2px 4px">历史记录</div>', unsafe_allow_html=True)
    with head_r:
        if st.button("折叠" if ss.hist_open else "展开", use_container_width=True, key="hist_toggle_btn"):
            ss.hist_open = not ss.hist_open
            st.rerun()

    if ss.hist_open:
        if ss.history:
            rows = []
            for i, h in enumerate(ss.history):
                cls = {"已找到": "s-ok", "查询中": "s-run"}.get(h["status"], "s-no")
                rows.append(f'<div class="hrow{" odd" if i % 2 == 0 else ""}"><span class="q">{html.escape(h["q"])}</span><span class="s {cls}">{h["status"]}</span></div>')
            st.markdown(f'<div class="hist">{"".join(rows)}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="hist"><div class="hrow odd"><span class="q" style="color:#a8b6c6">暂无搜索记录</span><span class="s s-no">—</span></div></div>', unsafe_allow_html=True)

    # 执行查询
    if ss.pending:
        pending = ss.pending
        ss.pending = None
        with st.spinner("正在比对照片区…"):
            try:
                result = run_search(pending, candidates, use_yolo=ss.use_yolo, top_k=ss.top_k, threshold=ss.threshold)
            except Exception as exc:
                result = {"q": pending, "items": [], "best": 0.0, "n_hit": 0, "n_lib": len(candidates), "status": "error", "msg": f"检索失败：{exc}"}
        ss.result = result
        for h in ss.history:
            if h["q"] == pending and h["status"] == "查询中":
                h["status"] = "已找到" if result["n_hit"] else ("出错" if result["status"] == "error" else "未找到")
                h["score"] = result["best"]
                break
        st.rerun()

    result = ss.result

    # 结果展示区
    if result and result["status"] in ("error", "empty") and not result["items"]:
        (st.error if result["status"] == "error" else st.info)(result["msg"])
        st.markdown(f'<div class="sec">备选图片<span class="count">图库 {result["n_lib"]} 张</span></div>', unsafe_allow_html=True)
        st.markdown(_grid_html([]), unsafe_allow_html=True)
    elif result and result["items"]:
        items = result["items"]
        for it in items: it["hit"] = it["score"] >= ss.threshold
        n_hit = sum(1 for it in items if it["hit"])

        st.markdown(f'<div class="sec">备选图片<span class="count">按匹配度排序 · 图库 {result["n_lib"]} 张</span></div>', unsafe_allow_html=True)
        st.markdown(_grid_html(items, ss.top_k, ss.show_full), unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="sec">备选图片<span class="count">图库 {len(candidates)} 张</span></div>', unsafe_allow_html=True)
        st.markdown(_grid_html([]), unsafe_allow_html=True)

    # 检索设置
    with st.expander("检索设置", expanded=False):
        ss.top_k = int(st.slider("返回图片数量", 4, 16, int(ss.top_k), 4))
        ss.threshold = float(st.slider("相似度阈值", 0.05, 0.60, float(ss.threshold), 0.01))
        ss.use_yolo = st.toggle("启用 YOLO 裁片", value=ss.use_yolo)
        ss.show_full = st.toggle("图块显示原始照片", value=ss.show_full)
        if st.button("清空历史"):
            ss.history, ss.result = [], None
            st.rerun()

if __name__ == "__main__":
    main()