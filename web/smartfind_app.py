"""
慧眼寻物 SmartFind —— 正式版图文检索界面（YOLO + CLIP 真实推理，无假数据）
=========================================================================
放置位置：D:/SmartFind/web/smartfind_app.py
运行方式：在项目根目录 D:/SmartFind 下执行
    streamlit run web/smartfind_app.py

依赖（缺一不可）：
    pip install streamlit torch torchvision ultralytics opencv-python ftfy regex
    pip install git+https://github.com/openai/CLIP.git     # 或 pip install clip-anytorch

【本文件做了什么】
    1. 直接调用队友已经写好的 YOLO.py / CLIP.py，不复制、不修改它们的逻辑：
         YOLODetector(model_path, conf_threshold).detect_and_crop(path) -> [PIL 子图]
         CLIPMatcher(threshold).match_images(text, [PIL 子图], top_k)   -> dict
    2. 把「待匹配照片」抽象成一个数据源适配层 load_candidates()：
         现在  = 读 data/ 下已有的照片
         将来  = 换成管理员上传的图片（文件 / 上传流 / SQLite blob），
                 只需要改这一处，检索、排序、界面全部不用动。
    3. 输出排序：一张照片命中多个候选框时取最高分，按分数从高到低铺满九宫格，
       不够的格子用虚线框占位。

【界面行为】
    · 顶部搜索框输入物品描述后按回车（或点右下隐藏的提交按钮）即开始检索；
    · 结果按匹配度从高到低排列，每张图右上角是相似度百分比，低于阈值的是灰色参考图；
    · 历史记录可折叠，状态分「查询中 / 已找到 / 未找到」；
    · 底部「检索设置」可临时关掉 YOLO（对比整图直接比对的效果）、改阈值、
      切换图块显示裁片还是原始照片、打开调试视图看 YOLO 裁了哪些区域、各得多少分。
    · 模型只会加载一次（st.cache_resource），第一次检索会慢几秒，之后秒回；
      终端里能看到 CLIP / YOLO 的加载日志。
    · 主题色取自 web/.streamlit/config.toml 的 primaryColor（#2f74d0），
      所以本文件请放在 web/ 目录下运行，蓝色主题才会生效。

【将来接管理员图片时改哪里】
    见下方 "★ 数据源适配层" 一节，已经把三种接入方式写成可直接调用的函数：
        load_candidates_from_uploads(files)   # Streamlit 文件上传控件
        load_candidates_from_dir(admin_dir)   # 管理员上传后统一存的目录
        load_candidates_from_sqlite(db_path)  # 管理员提交进 SQLite
    然后在 load_candidates() 里 return 对应的那个函数即可。
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
# 路径与配置：全部基于文件自身位置推导，无论从哪个目录启动都能找到模型和图片
# ============================================================================
_HERE = Path(__file__).resolve().parent
# 放在 web/ 下时根目录是上一级；直接放在项目根目录时根目录就是自己
ROOT = _HERE.parent if (_HERE.parent / "YOLO.py").exists() else _HERE

if str(ROOT) not in sys.path:            # 保证 import YOLO / CLIP 能找到
    sys.path.insert(0, str(ROOT))

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# —— 现在：直接读项目里已有的照片做验证 ——
LIBRARY_DIRS = [
    ROOT / "data",                       # data/1.jpg 2.jpg 3.jpg bottle+laptop.jpg bus.jpg
    # ROOT / "web" / "test_images",      # 想扩图库时把这一行放开即可
]

# —— 将来：管理员提交的图片统一落到这个目录（现在不存在也不影响运行）——
ADMIN_UPLOAD_DIR = ROOT / "web" / "admin_uploads"

YOLO_WEIGHTS = ROOT / "models" / "yolov8n.pt"
YOLO_CONF = 0.15                # 监控检索场景：宁可多框，交给 CLIP 去筛
DEFAULT_THRESHOLD = 0.25        # 相似度判定阈值（CLIP ViT-B/32 建议 0.2~0.3）
DEFAULT_TOPK = 8                # 九宫格：4 列 x 2 行
GRID_SLOTS = 8                  # 不足时补虚线占位框的数量
LOGO_PATHS = [ROOT / "web" / "logo.png", ROOT / "logo.png", _HERE / "logo.png"]


# ============================================================================
# ★ 数据源适配层：所有「待匹配照片」都从这里进系统
# ============================================================================
@dataclass
class Candidate:
    """一张待匹配的照片（将来也可以是一条管理员登记记录）。"""
    key: str                        # 唯一标识：文件路径 / 上传文件名 / 数据库主键
    image: Image.Image              # RGB 原图
    label: str                      # 展示名称：文件名 或 管理员填的物品名
    source: str = "file"            # 来源标记：file / admin_upload / db
    path: Optional[Path] = None     # 有实体文件时给 YOLO 直接读（更快）
    meta: dict = field(default_factory=dict)


def scan_images(directory: Path) -> List[Path]:
    """遍历目录下的图片文件，跳过损坏 / 空文件。"""
    if not directory or not directory.is_dir():
        return []
    found: List[Path] = []
    for ext in IMAGE_EXTS:
        found.extend(glob.glob(str(directory / f"*{ext}")))
        found.extend(glob.glob(str(directory / f"*{ext.upper()}")))
    out, seen = [], set()
    for p in sorted(found):
        if p in seen:
            continue
        seen.add(p)
        try:
            if os.path.getsize(p) > 0:
                out.append(Path(p))
        except OSError:
            continue
    return out


def candidate_from_path(path: Path, label: Optional[str] = None, source: str = "file") -> Optional[Candidate]:
    """文件 -> Candidate（读不出来就返回 None，保证界面不会崩）。"""
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
    """方式一：管理员把图片存到统一目录，系统扫目录即可。"""
    items = []
    for p in scan_images(directory):
        c = candidate_from_path(p, source=source)
        if c:
            items.append(c)
    return items


def load_candidates_from_uploads(files: Iterable) -> List[Candidate]:
    """方式二：直接把 Streamlit 的 st.file_uploader 结果转成 Candidate。

    管理员页面里这样用：
        files = st.file_uploader("上传监控照片", accept_multiple_files=True)
        cands = load_candidates_from_uploads(files)
    之后调用 run_search(query, cands, ...) 就能对刚上传的图做匹配。
    """
    items = []
    for f in files or []:
        try:
            img = Image.open(f).convert("RGB")
        except Exception:
            continue
        items.append(Candidate(
            key=getattr(f, "name", str(id(f))),
            image=img,
            label=getattr(f, "name", "上传图片"),
            source="admin_upload",
            path=None,               # 没有落盘：YOLO 直接吃 PIL 图
        ))
    return items


def load_candidates_from_sqlite(db_path: Path, table: str = "lost_item") -> List[Candidate]:
    """方式三：管理员登记信息进 SQLite（图片存 BLOB 或存路径）。

    表结构参考需求文档：lost_item(id, name, desc, image_path / image_blob, created_at)
    这里写成可直接启用的实现，表不存在时返回空列表，不会影响页面。
    """
    import sqlite3

    if not Path(db_path).exists():
        return []
    items: List[Candidate] = []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        cols = set(rows[0].keys()) if rows else set()
        for row in rows:
            img = None
            if "image_blob" in cols and row["image_blob"]:
                img = Image.open(io.BytesIO(row["image_blob"])).convert("RGB")
            elif "image_path" in cols and row["image_path"]:
                p = Path(row["image_path"])
                p = p if p.is_absolute() else ROOT / p
                if p.exists():
                    img = Image.open(p).convert("RGB")
            if img is None:
                continue
            items.append(Candidate(
                key=f"db:{row['id']}",
                image=img,
                label=row["name"] if "name" in cols and row["name"] else f"记录 {row['id']}",
                source="db",
                meta={"created_at": row["created_at"] if "created_at" in cols else ""},
            ))
        conn.close()
    except Exception:
        return []
    return items


def load_candidates() -> List[Candidate]:
    """【当前生效的数据源】项目里已有的照片。

    ★ 将来接入管理员图片时，把这里换成下面任意一行即可，其它代码零改动：
        return load_candidates_from_dir(ADMIN_UPLOAD_DIR)
        return load_candidates_from_sqlite(ROOT / "smartfind.db")
        return load_candidates_from_uploads(st.session_state.get("admin_files"))
    """
    items: List[Candidate] = []
    for d in LIBRARY_DIRS:
        items.extend(load_candidates_from_dir(d, source="file"))
    # 管理员目录存在时自动并入（现在目录不存在，什么都不会发生）
    if ADMIN_UPLOAD_DIR.is_dir():
        items.extend(load_candidates_from_dir(ADMIN_UPLOAD_DIR, source="admin_upload"))
    return items


# ============================================================================
# 算法层：YOLO 裁片 -> CLIP 打分（完全复用队友的类，不重写推理逻辑）
# ============================================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    """模型只加载一次（Streamlit 全局缓存），第二次查询秒回。"""
    from YOLO import YOLODetector
    from CLIP import CLIPMatcher

    detector = YOLODetector(model_path=str(YOLO_WEIGHTS), conf_threshold=YOLO_CONF)
    matcher = CLIPMatcher(threshold=DEFAULT_THRESHOLD)
    return detector, matcher


_CROP_CACHE: dict = {}          # 同一张图的 YOLO 裁片结果复用，避免重复推理


def crops_for(detector, cand: Candidate) -> List[Image.Image]:
    """对一张照片做 YOLO 检测，返回候选子图（结果按 (图, 是否检测过) 缓存）。"""
    ck = (cand.key, cand.meta.get("mtime"), cand.meta.get("size"))
    if ck in _CROP_CACHE:
        return _CROP_CACHE[ck]
    target = str(cand.path) if cand.path else cand.image   # 有路径喂路径，没有喂 PIL
    crops = detector.detect_and_crop(target) or []
    _CROP_CACHE[ck] = crops
    return crops


def run_search(query: str, candidates: List[Candidate], *, use_yolo: bool = True,
               top_k: int = DEFAULT_TOPK, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """完整检索流程：YOLO 抽候选区域 → CLIP 批量打分 → 按图聚合 → 排序。

    关键点：
      · 所有照片的所有裁片拼成**一次** batch 送进 CLIP，速度最快；
      · 一张照片命中多个裁片时取最高分作为这张照片的分数；
      · YOLO 什么都没框到时用整张图兜底——保证「照片本身」也能参与匹配。
    """
    detector, matcher = get_engine()
    matcher.threshold = threshold            # 阈值随界面滑块实时生效

    pool: List[Image.Image] = []             # 所有裁片（送进 CLIP 的大 batch）
    owner: List[str] = []                    # 每个裁片属于哪张照片（key）
    info: dict = {}                          # key -> 这张照片的裁片情况

    for cand in candidates:
        try:
            crops = crops_for(detector, cand) if use_yolo else []
        except Exception:
            crops = []
        detected = bool(crops)
        if not crops:                        # 兜底：整图
            crops = [cand.image]
        info[cand.key] = {"cand": cand, "n_crops": len(crops), "detected": detected}
        for crop in crops:
            pool.append(crop)
            owner.append(cand.key)

    if not pool:
        return {"q": query, "items": [], "best": 0.0, "n_hit": 0,
                "n_lib": len(candidates), "status": "empty",
                "msg": "图库为空：没有可用的照片，请检查 LIBRARY_DIRS，或让管理员先上传图片"}

    res = matcher.match_images(query, pool, top_k=len(pool))
    if res["status"] != "success":
        return {"q": query, "items": [], "best": 0.0, "n_hit": 0,
                "n_lib": len(candidates), "msg": res["message"], "status": "error"}

    # 裁片 -> 照片 的反查表（match_images 原样返回传入的 PIL 对象，用 id 对齐最稳）
    idx_by_id = {id(crop): owner[i] for i, crop in enumerate(pool)}
    grouped: dict = {}
    for crop, score in res["top_k_results"]:
        key = idx_by_id.get(id(crop))
        if key is None:
            continue
        grouped.setdefault(key, []).append((crop, float(score)))

    items = []
    for key, hits in grouped.items():
        hits.sort(key=lambda x: x[1], reverse=True)
        best_crop, best_score = hits[0]
        cand = info[key]["cand"]
        items.append({
            "key": key,
            "label": cand.label,
            "source": cand.source,
            "score": best_score,
            "hit": best_score >= threshold,
            "thumb": best_crop,
            "full": cand.image,
            "n_crops": info[key]["n_crops"],
            "detected": info[key]["detected"],
            "debug": hits[:4],               # 调试视图：这张图得分最高的几个裁片
        })

    items.sort(key=lambda r: r["score"], reverse=True)
    best = items[0]["score"] if items else 0.0
    n_hit = sum(1 for r in items if r["hit"])
    return {
        "q": query,
        "items": items[:top_k],              # 只铺满九宫格
        "best": best,
        "n_hit": n_hit,
        "n_lib": len(candidates),
        "msg": f"命中 {n_hit} 张" if n_hit else "未找到达到阈值的照片",
        "status": "ok",
    }


# ============================================================================
# 展示层：样式 / 图片编码 / HTML 小组件
# ============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');

/* ---------- 背景：浅蓝渐变 + 左右柔光，和设计稿一致 ---------- */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background:
        radial-gradient(900px 520px at 6% 4%, #ffffff 0%, rgba(255,255,255,0) 62%),
        radial-gradient(820px 480px at 92% 96%, #ffffff 0%, rgba(255,255,255,0) 58%),
        linear-gradient(158deg, #eef8ff 0%, #dcefff 38%, #bfe1fb 100%) !important;
    font-family: "Noto Sans SC", "HarmonyOS Sans SC", "PingFang SC",
                 "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
}

/* ---------- 藏掉 Streamlit 原生外壳 ---------- */
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], [data-testid="InputInstructions"] { display: none !important; }
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

/* ---------- 搜索胶囊：外框画在容器上，input 本身透明 ---------- */
[data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; }
.st-key-search [data-testid="stTextInput"] :is([data-baseweb="input"], .react-aria-TextField > div),
.st-key-search [data-testid="stTextInput"] :is([data-testid="stTextInputRootElement"]) {
    position: relative !important;
    height: 56px !important;
    display: flex !important; align-items: center !important;
    border-radius: 30px !important;
    border: 1.6px solid #8ab4e0 !important;
    background-color: #ffffff !important;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%233f7bc4' stroke-width='1.9' stroke-linecap='round'><circle cx='11' cy='11' r='7'/><line x1='16.4' y1='16.4' x2='21' y2='21'/></svg>");
    background-repeat: no-repeat; background-position: 20px center; background-size: 19px 19px;
    box-shadow: 0 4px 18px rgba(47,116,208,.07) !important;
    transition: border-color .2s, box-shadow .2s;
}
.st-key-search [data-testid="stTextInput"] :is([data-baseweb="input"], .react-aria-TextField > div):focus-within {
    border-color: #2f74d0 !important;
    box-shadow: 0 0 0 5px rgba(47,116,208,.10) !important;
}
.st-key-search input {
    height: 100% !important; border: none !important; outline: none !important;
    background: transparent !important; box-shadow: none !important;
    padding: 0 22px 0 50px !important;
    font-size: 15px !important; color: #24405e !important;
}
.st-key-search input::placeholder { color: #92a8c1 !important; }
.st-key-search [data-testid="stFormSubmitButton"] {
    position: absolute !important; left: -9999px; width: 1px; height: 1px; opacity: 0;
}
[data-testid="stTextInput"] label { display: none !important; }

/* ---------- 分区标题 ---------- */
.sec {
    display: flex; justify-content: space-between; align-items: center;
    margin: 20px 2px 8px; font-size: 13.5px; font-weight: 600;
    color: #4a6d94; letter-spacing: .4px;
}
.sec .count { font-weight: 400; color: #93a9c2; font-size: 12.5px; }

/* ---------- 历史记录 ---------- */
.hist { display: flex; flex-direction: column; gap: 2px; }
.hrow {
    display: flex; justify-content: space-between; align-items: center;
    padding: 11px 16px; font-size: 14.5px; color: #2c405c; border-radius: 8px;
}
.hrow.odd { background: #f1f8ff; }
.hrow .q { max-width: 76%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.hrow .s { font-size: 13px; font-weight: 500; white-space: nowrap; }
.s-ok { color: #17a05e; } .s-run { color: #2f74d0; } .s-no { color: #98a6b8; }
.st-key-hist_toggle button {
    background: transparent !important; border: none !important; padding: 0 !important;
    color: #6b8bad !important; font-size: 12.5px !important; font-weight: 400 !important;
    height: 20px !important; min-height: 20px !important; line-height: 20px !important;
    box-shadow: none !important;
}
.st-key-hist_toggle button:hover { color: #2f74d0 !important; }

/* ---------- 结果九宫格 ---------- */
.grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 4px; }
.tile {
    position: relative; border-radius: 13px; overflow: hidden;
    aspect-ratio: 1 / 1; background: #e8e8e8;
    display: flex; align-items: center; justify-content: center;
    color: #b3b3b3; font-size: 12.5px; letter-spacing: .5px;
    animation: pop .34s cubic-bezier(.2,.8,.3,1) both;
}
@keyframes pop { from { opacity: 0; transform: translateY(8px) scale(.97); } to { opacity: 1; transform: none; } }
.tile img { width: 100%; height: 100%; object-fit: cover; display: block; transition: transform .45s ease; }
.tile:hover img { transform: scale(1.07); }
.tile.ref img { filter: saturate(.75) brightness(.97); }        /* 未达阈值：视觉弱化 */
.tile .chip {
    position: absolute; top: 6px; right: 6px; z-index: 2;
    font-size: 10.5px; font-weight: 700; padding: 2px 7px; border-radius: 999px;
    background: rgba(47,116,208,.94); color: #fff; letter-spacing: .2px;
}
.tile .chip.ref { background: rgba(255,255,255,.92); color: #7e93a9; }
.tile .cap {
    position: absolute; left: 0; right: 0; bottom: 0; z-index: 2;
    padding: 16px 8px 6px; font-size: 10.5px; color: #fff;
    background: linear-gradient(transparent, rgba(0,0,0,.56));
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    opacity: 0; transition: opacity .25s;
}
.tile:hover .cap { opacity: 1; }
/* 虚线占位框 */
.tile.empty {
    background: #fbfdff;
    border: 1.5px dashed #c5d8ea;
    color: #b9c9d9;
}

/* ---------- 结果概览 ---------- */
.sum { margin-top: 12px; font-size: 12.5px; color: #8ea4bd; display: flex; justify-content: space-between; gap: 10px; }
.sum b { color: #2f74d0; font-weight: 700; }
.pill { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 11.5px; font-weight: 600; }
.pill.ok { background: #e8f8ef; color: #17a05e; }
.pill.no { background: #f2f5f9; color: #8496a8; }
/* ---------- 空状态提示 ---------- */
.hint { text-align: center; color: #a3b6c9; font-size: 12.5px; padding: 14px 0 6px; }

/* ---------- 调试视图 ---------- */
.dbgwrap { display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 10px; }
.dbg { position: relative; width: 82px; height: 82px; border-radius: 9px; overflow: hidden; background: #eef3f8; }
.dbg img { width: 100%; height: 100%; object-fit: cover; }
.dbg span {
    position: absolute; left: 0; right: 0; bottom: 0; text-align: center;
    font-size: 10px; color: #fff; background: rgba(0,0,0,.5); padding: 1px 0;
}

/* ---------- 折叠面板（检索设置 / 调试视图） ---------- */
[data-testid="stExpander"] {
    border: 1px solid #eaf2fb !important; border-radius: 14px !important;
    background: #fcfdff; margin-top: 14px;
}
[data-testid="stExpander"] details > summary {
    padding: 9px 14px !important; font-size: 12.5px !important; color: #6b8bad !important;
}
[data-testid="stExpander"] details > summary:hover { color: #2f74d0 !important; }
[data-testid="stExpander"] details > summary p { font-size: 12.5px !important; }

[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { color: #2f74d0 !important; }
[data-testid="stExpander"] [data-testid="stSlider"] { padding-top: 10px; }
/* 万一主题配置没被读到，也强制用品牌蓝（web/.streamlit/config.toml 已设 primaryColor） */
[data-testid="stCheckbox"][data-selected="true"] > label > div:first-of-type { background-color: #2f74d0 !important; }
[data-testid="stAlert"] { border-radius: 12px; font-size: 13px; }
</style>
"""


@st.cache_data(show_spinner=False)
def _logo_b64(path_str: str, max_w: int = 520) -> Optional[str]:
    """读 LOGO：裁掉四周白边 + 缩放 + 转 base64（自定义 HTML 才能引用本地图）。"""
    if not os.path.exists(path_str):
        return None
    try:
        im = Image.open(path_str).convert("RGB")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        diff = ImageChops.difference(im, bg).convert("L").point(lambda v: 255 if v > 12 else 0)
        box = diff.getbbox()
        if box:
            im = im.crop(box)
        if im.width > max_w:
            im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _b64_image(img: Image.Image, max_side: int = 460, quality: int = 82) -> str:
    """PIL 图 -> base64 JPEG（先缩放，避免大图拖慢页面）。"""
    im = img.convert("RGB")
    w, h = im.size
    scale = max_side / max(w, h)
    if scale < 1:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _tile_html(item: dict, show_full: bool = False) -> str:
    """一张结果图：默认显示 YOLO 匹配到的裁片，可切换成原始照片。"""
    src = _b64_image(item["full"] if show_full else item["thumb"])
    pct = round(item["score"] * 100)
    cls = "" if item["hit"] else " ref"
    return (
        f'<div class="tile{cls}">'
        f'<img src="data:image/jpeg;base64,{src}" alt="">'
        f'<span class="chip{cls}">{pct}%</span>'
        f'<span class="cap">{html.escape(str(item["label"]))}</span>'
        f'</div>'
    )


_PLACEHOLDER = '<div class="tile empty">备选图片</div>'


def _grid_html(items: List[dict], slots: int = GRID_SLOTS, show_full: bool = False) -> str:
    """九宫格：有结果就出图，不够的格子用虚线框占位。"""
    tiles = [_tile_html(it, show_full) for it in items[:slots]]
    tiles.extend([_PLACEHOLDER] * max(0, slots - len(tiles)))
    return f'<div class="grid">{"".join(tiles)}</div>'


def _debug_html(items: List[dict]) -> str:
    blocks = []
    for it in items:
        chips = "".join(
            f'<div class="dbg"><img src="data:image/jpeg;base64,{_b64_image(img, 160, 75)}">'
            f'<span>{score * 100:.1f}</span></div>'
            for img, score in it["debug"]
        )
        blocks.append(
            f'<div style="font-size:12px;color:#6b8bad;margin:8px 0 4px">'
            f'{html.escape(str(it["label"]))} · 候选框 {it["n_crops"]} 个'
            f'{"（整图参与匹配）" if not it["detected"] else ""}</div>'
            f'<div class="dbgwrap">{chips}</div>'
        )
    return "".join(blocks)


# ============================================================================
# 页面装配
# ============================================================================
def main() -> None:
    # st.set_page_config(page_title="慧眼寻物 SmartFind", page_icon="🔍", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)

    ss = st.session_state
    ss.setdefault("history", [])                      # [{"q", "status", "score"}]
    ss.setdefault("result", None)
    ss.setdefault("pending", None)
    ss.setdefault("hist_open", True)
    ss.setdefault("top_k", DEFAULT_TOPK)
    ss.setdefault("threshold", DEFAULT_THRESHOLD)
    ss.setdefault("use_yolo", True)
    ss.setdefault("show_full", False)
    ss.setdefault("show_debug", False)

    # ---------- 顶部：LOGO + 头像 ----------
    logo_b64 = None
    for p in LOGO_PATHS:
        logo_b64 = _logo_b64(str(p))
        if logo_b64:
            break
    if logo_b64:
        st.markdown(f'<div class="brand"><img src="data:image/png;base64,{logo_b64}" '
                    f'alt="慧眼寻物 SmartFind"></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="brand"><span class="txt">慧眼寻物 SmartFind</span></div>',
                    unsafe_allow_html=True)

    st.markdown("""
    <div class="avatar">
      <svg width="29" height="29" viewBox="0 0 24 24" fill="none" stroke="#8b98a8" stroke-width="1.6">
        <circle cx="12" cy="8.2" r="3.6"/><path d="M4.6 20c1.2-3.9 4-5.8 7.4-5.8s6.2 1.9 7.4 5.8"/>
      </svg>
    </div>
    """, unsafe_allow_html=True)

    # ---------- 搜索栏（回车即搜） ----------
    with st.container(key="search"):
        with st.form("search_form", clear_on_submit=False):
            query = st.text_input("q", value="", label_visibility="collapsed",
                                  placeholder="输入名称、特征，搜索失物")
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

    # ---------- 历史记录 ----------
    head_l, head_r = st.columns([4, 1], vertical_alignment="center")
    with head_l:
        st.markdown('<div class="sec" style="margin:18px 2px 4px">历史记录</div>',
                    unsafe_allow_html=True)
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
                    f'<span class="q">{html.escape(h["q"])}</span>'
                    f'<span class="s {cls}">{h["status"]}</span></div>'
                )
            st.markdown(f'<div class="hist">{"".join(rows)}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="hist"><div class="hrow odd">'
                        '<span class="q" style="color:#a8b6c6">暂无搜索记录</span>'
                        '<span class="s s-no">—</span></div></div>', unsafe_allow_html=True)

    # ---------- 执行待处理查询（放在历史记录之后，用户能看到“查询中”） ----------
    if ss.pending:
        pending = ss.pending
        ss.pending = None
        with st.spinner("正在比对照片区…"):
            try:
                result = run_search(pending, candidates, use_yolo=ss.use_yolo,
                                    top_k=ss.top_k, threshold=ss.threshold)
            except Exception as exc:               # 模型缺失 / 显存不足等都不能白屏
                result = {"q": pending, "items": [], "best": 0.0, "n_hit": 0,
                          "n_lib": len(candidates), "status": "error",
                          "msg": f"检索失败：{exc}"}
        ss.result = result
        for h in ss.history:
            if h["q"] == pending and h["status"] == "查询中":
                h["status"] = "已找到" if result["n_hit"] else (
                    "出错" if result["status"] == "error" else "未找到")
                h["score"] = result["best"]
                break
        st.rerun()

    result = ss.result

    # ---------- 结果区 ----------
    if result and result["status"] in ("error", "empty") and not result["items"]:
        tip = result["msg"]
        (st.error if result["status"] == "error" else st.info)(tip)
        st.markdown(f'<div class="sec">备选图片<span class="count">图库 {result["n_lib"]} 张</span></div>',
                    unsafe_allow_html=True)
        st.markdown(_grid_html([]), unsafe_allow_html=True)
    elif result and result["items"]:
        # 命中与否按「当前」阈值实时判定：拖动阈值滑块，灰色/蓝色立刻跟着变
        items = result["items"]
        for it in items:
            it["hit"] = it["score"] >= ss.threshold
        n_hit = sum(1 for it in items if it["hit"])

        st.markdown(
            f'<div class="sec">备选图片<span class="count">按匹配度排序 · '
            f'图库 {result["n_lib"]} 张</span></div>', unsafe_allow_html=True)
        st.markdown(_grid_html(items, ss.top_k, ss.show_full), unsafe_allow_html=True)

        pill = '<span class="pill ok">已找到</span>' if n_hit else '<span class="pill no">未达阈值</span>'
        extra = "（灰色为相似度偏低的参考画面）" if n_hit < len(items) else ""
        st.markdown(
            f'<div class="sum"><span>{pill} 「{html.escape(result["q"])}」命中 '
            f'<b>{n_hit}</b> 张 {extra}</span>'
            f'<span>最高相似度 <b>{result["best"] * 100:.1f}%</b></span></div>',
            unsafe_allow_html=True)

        if ss.show_debug:
            with st.expander("调试视图：YOLO 裁片与 CLIP 打分", expanded=True):
                st.caption("每张小图是 YOLO 从原照片里裁出的候选区域，下方数字为与输入描述的 CLIP 相似度（×100）。")
                st.markdown(_debug_html(items[:4]), unsafe_allow_html=True)
    else:
        st.markdown('<div class="sec">备选图片<span class="count">'
                    f'图库 {len(candidates)} 张</span></div>', unsafe_allow_html=True)
        st.markdown(_grid_html([]), unsafe_allow_html=True)
        st.markdown('<div class="hint">输入物品名称或特征开始查找，例如「红色双肩包」「黄色保温杯」</div>',
                    unsafe_allow_html=True)

    # ---------- 检索设置（默认收起，展开后可调 YOLO / 阈值，用于对比测试） ----------
    lib_dirs = "、".join(d.name for d in LIBRARY_DIRS)
    with st.expander(f"检索设置 · 图库 {len(candidates)} 张（{lib_dirs}）", expanded=False):
        st.caption("YOLO 负责从照片中裁出候选区域，CLIP 负责计算裁片与文字描述的相似度；"
                   "关掉 YOLO 即可对比「整图直接比对」的效果。")
        ss.top_k = int(st.slider("返回图片数量", 4, 16, int(ss.top_k), 4))
        ss.threshold = float(st.slider("相似度阈值（低于该值视为未命中）",
                                       0.05, 0.60, float(ss.threshold), 0.01))
        ss.use_yolo = st.toggle("启用 YOLO 裁片", value=ss.use_yolo,
                                help="关闭后整张照片直接和文字比对（改完需重新搜索）")
        ss.show_full = st.toggle("图块显示原始照片", value=ss.show_full,
                                 help="关闭则显示 YOLO 匹配到的裁片")
        ss.show_debug = st.toggle("显示调试视图", value=ss.show_debug,
                                  help="展开显示每张照片的候选裁片与得分")

        b1, b2 = st.columns(2)
        if b1.button("重新扫描图库", use_container_width=True):
            _CROP_CACHE.clear()         # 清掉 YOLO 裁片缓存，新加的照片立刻生效
            st.rerun()
        if b2.button("清空历史", use_container_width=True):
            ss.history, ss.result = [], None
            st.rerun()


if __name__ == "__main__":
    main()
