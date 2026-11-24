# ============================================================================
# 慧眼寻物 SmartFind —— 统一前端入口（三合一，功能完整保留）
# ============================================================================
# 本文件由原来的三个前端合并而来，功能一个不少：
#   · web/app.py          —— Streamlit 简洁版（MOCK_MODE 开关、Mac 强制 CPU、缩略图裁剪）
#   · web/smartfind_app.py—— Streamlit 正式版（数据源适配层、九宫格、调试视图、检索设置）
#   · web/smartfind_web.py—— Flask 独立网页（中译英词典、管理员上传、独立 HTML 页面）
# 三个旧文件已备份在 web/_legacy/ 目录，平时不用管。
#
# 【两种启动方式】按你的需要选一种即可，不要再去找第二个 app.py：
#   1) Streamlit 界面（推荐，功能最全）：
#        在项目根目录执行：  streamlit run web/app.py
#        浏览器打开终端显示的地址（默认 http://localhost:8501）
#   2) Flask 独立网页（自带管理员上传入口）：
#        在项目根目录执行：  .venv/bin/python web/app.py
#        或明确指定：        .venv/bin/python web/app.py flask
#        浏览器打开 http://127.0.0.1:5050
#        （5000 端口常被 macOS AirPlay 占用，需要的话用 SMARTFIND_PORT=6001 改端口）
#
#   演示模式（不加载模型、秒出假数据）：
#        streamlit run web/app.py -- --mock    或    SMARTFIND_MOCK=1 python web/app.py
# ============================================================================

# ---- 环境补丁：必须放在所有 import 之前，解决 Mac OpenMP 冲突 ----
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import base64
import glob
import html
import io
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from PIL import Image, ImageChops

# ---- 路径：全部基于文件自身位置推导，从任何目录启动都能找到模型和图片 ----
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if (_HERE.parent / "YOLO.py").exists() else _HERE
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- Streamlit 是可选的：Flask 模式下没装也能跑 ----
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    st = None
    _HAS_STREAMLIT = False

# torch / cv2 是真实识别时才需要的重型依赖：这里做容错导入，
# 缺依赖时页面仍能启动（可用 MOCK 演示模式），真正检索时才在 _build_engine() 里报错提示。
try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    torch = None
try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    cv2 = None

# ============================================================================
# ★★★ 配置区 ★★★
# ============================================================================
MOCK_MODE = (os.environ.get("SMARTFIND_MOCK", "0") == "1") or ("--mock" in sys.argv)
IMAGE_DIRS = [ROOT / "web" / "test_images", ROOT / "data"]   # 图库目录（能被搜索的照片）
ADMIN_UPLOAD_DIR = ROOT / "web" / "admin_uploads"            # 管理员上传落盘目录
LOGO_PATHS = [ROOT / "web" / "logo.png", ROOT / "logo.png", _HERE / "logo.png"]
YOLO_WEIGHTS = ROOT / "models" / "yolov8n.pt"
YOLO_CONF = 0.15                # 监控检索场景：宁可多框，交给 CLIP 去筛
DEFAULT_THRESHOLD = 0.25        # 相似度判定阈值（CLIP score 为 0~1）
DEFAULT_TOPK = 8                # 九宫格 4×2
GRID_SLOTS = 8
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# 端口可用环境变量改：macOS 的 AirPlay 常占用 5000，故默认用 5050
API_HOST = os.environ.get("SMARTFIND_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("SMARTFIND_PORT", "5050"))

# ============================================================================
# 中译英词典（来自 Flask 版）：CLIP 官方模型中文弱，输入含中文时自动追加英文提示
# 长词优先 + 命中即消耗 + 同义词去重，全程离线
# ============================================================================
ZH2EN = {
    # ---------- 箱包 ----------
    "背包": "backpack", "书包": "backpack", "双肩包": "backpack", "挎包": "shoulder bag",
    "单肩包": "shoulder bag", "手提包": "handbag", "手拎包": "handbag", "公文包": "briefcase",
    "腰包": "waist bag", "收纳包": "pouch", "化妆包": "makeup bag", "旅行包": "travel bag",
    "行李箱": "suitcase", "拉杆箱": "suitcase", "旅行箱": "suitcase", "箱子": "box",
    "钱包": "wallet", "钱夹": "wallet", "卡包": "card holder", "笔袋": "pencil case",
    "文具盒": "pencil case", "购物袋": "shopping bag", "纸袋": "paper bag", "布袋": "cloth bag",
    # ---------- 电子数码 ----------
    "手机": "mobile phone", "电话": "phone", "苹果手机": "iphone", "华为手机": "huawei phone",
    "平板": "tablet", "笔记本电脑": "laptop", "笔记本": "notebook", "电脑": "computer",
    "耳机": "earphones", "蓝牙耳机": "bluetooth earphones", "有线耳机": "wired earphones",
    "充电宝": "power bank", "充电器": "charger", "数据线": "cable", "充电线": "charging cable",
    "鼠标": "mouse", "键盘": "keyboard", "U盘": "usb drive", "优盘": "usb drive",
    "移动硬盘": "external hard drive", "硬盘": "hard drive", "相机": "camera",
    "单反": "dslr camera", "摄像机": "video camera", "手表": "watch", "智能手表": "smart watch",
    "手环": "fitness band", "游戏机": "game console", "对讲机": "walkie talkie",
    "录音笔": "voice recorder", "计算器": "calculator", "手机壳": "phone case",
    "保护壳": "protective case", "钢化膜": "screen protector", "自拍杆": "selfie stick",
    "三脚架": "tripod", "麦克风": "microphone", "音箱": "speaker", "音响": "speaker",
    "投影仪": "projector", "路由器": "router", "电池": "battery", "灯泡": "light bulb",
    "手电筒": "flashlight", "手电": "flashlight", "台灯": "desk lamp", "风扇": "fan",
    "吹风机": "hair dryer", "电动牙刷": "electric toothbrush", "剃须刀": "razor", "充电头": "charger",
    # ---------- 服饰穿戴 ----------
    "帽子": "hat", "鸭舌帽": "cap", "棒球帽": "baseball cap", "毛线帽": "beanie",
    "围巾": "scarf", "手套": "gloves", "口罩": "face mask", "眼镜": "glasses",
    "墨镜": "sunglasses", "太阳镜": "sunglasses", "隐形眼镜": "contact lenses",
    "衣服": "clothes", "外套": "jacket", "夹克": "jacket", "卫衣": "hoodie",
    "毛衣": "sweater", "衬衫": "shirt", "裤子": "pants", "裙子": "skirt",
    "鞋子": "shoes", "运动鞋": "sneakers", "帆布鞋": "canvas shoes", "拖鞋": "slippers",
    "袜子": "socks", "领带": "tie", "皮带": "belt", "腰带": "belt",
    "睡衣": "pajamas", "泳衣": "swimsuit", "雨衣": "raincoat", "校服": "school uniform",
    # ---------- 文具书籍 ----------
    "笔": "pen", "钢笔": "fountain pen", "圆珠笔": "ballpoint pen", "铅笔": "pencil",
    "中性笔": "gel pen", "马克笔": "marker", "荧光笔": "highlighter", "毛笔": "brush pen",
    "书": "book", "课本": "textbook", "字典": "dictionary", "词典": "dictionary",
    "尺子": "ruler", "橡皮": "eraser", "胶带": "tape", "订书机": "stapler",
    "剪刀": "scissors", "文件夹": "folder", "文件袋": "file bag", "画板": "drawing board",
    "画笔": "paint brush", "颜料": "paint", "作业本": "notebook",
    # ---------- 杯瓶容器 ----------
    "水杯": "cup", "杯子": "cup", "保温杯": "thermos cup", "水壶": "water bottle",
    "保温瓶": "thermos", "瓶子": "bottle", "瓶": "bottle", "矿泉水瓶": "water bottle",
    "茶杯": "teacup", "马克杯": "mug", "饭盒": "lunch box", "便当盒": "lunch box",
    # ---------- 钥匙证件卡 ----------
    "钥匙": "keys", "钥匙扣": "keychain", "门禁卡": "access card", "公交卡": "bus card",
    "地铁卡": "metro card", "饭卡": "meal card", "校园卡": "campus card",
    "银行卡": "bank card", "身份证": "id card", "驾驶证": "driver license",
    "会员卡": "membership card", "卡片": "card",
    # ---------- 雨具户外 ----------
    "伞": "umbrella", "雨伞": "umbrella", "遮阳伞": "parasol", "帐篷": "tent",
    "睡袋": "sleeping bag", "登山杖": "hiking pole", "望远镜": "binoculars",
    "指南针": "compass", "打火机": "lighter",
    # ---------- 颜色 ----------
    "紫色": "purple", "白色": "white", "黑色": "black", "红色": "red",
    "蓝色": "blue", "绿色": "green", "黄色": "yellow", "粉色": "pink",
    "灰色": "gray", "橙色": "orange", "棕色": "brown", "咖啡色": "brown",
    "银色": "silver", "金色": "gold", "透明": "transparent", "藏青色": "navy blue",
    # ---------- 材质图案 ----------
    "皮革": "leather", "真皮": "leather", "帆布": "canvas", "尼龙": "nylon",
    "塑料": "plastic", "金属": "metal", "木质": "wooden", "玻璃": "glass",
    "条纹": "striped", "格子": "plaid", "卡通": "cartoon", "印花": "floral print",
    # ---------- 人物动物 ----------
    "人": "person", "男人": "man", "女人": "woman", "小孩": "child", "儿童": "child",
    "老人": "elderly person", "学生": "student", "老师": "teacher", "警察": "police officer",
    "保安": "security guard", "猫": "cat", "小猫": "cat", "狗": "dog", "小狗": "dog",
    "兔子": "rabbit", "鸟": "bird", "鱼": "fish", "乌龟": "turtle", "仓鼠": "hamster",
    # ---------- 交通工具 ----------
    "公交车": "bus", "巴士": "bus", "汽车": "car", "轿车": "car", "出租车": "taxi",
    "自行车": "bicycle", "电动车": "electric scooter", "摩托车": "motorcycle",
    "地铁": "subway", "火车": "train", "高铁": "high-speed train", "飞机": "airplane",
    "三轮车": "tricycle", "滑板车": "scooter", "平衡车": "hoverboard",
    # ---------- 运动娱乐 / 其他 ----------
    "篮球": "basketball", "足球": "soccer ball", "排球": "volleyball",
    "羽毛球": "badminton racket", "乒乓球": "ping pong paddle", "网球": "tennis racket",
    "玩具": "toy", "娃娃": "doll", "玩偶": "stuffed toy", "积木": "building blocks",
    "拼图": "puzzle", "瑜伽垫": "yoga mat", "跳绳": "jump rope", "哑铃": "dumbbell",
    "镜子": "mirror", "梳子": "comb", "指甲刀": "nail clipper", "药": "medicine",
    "药瓶": "medicine bottle", "纸巾": "tissue", "快递": "package", "包裹": "package",
    "行李": "luggage", "纸箱": "cardboard box", "盒子": "box", "袋子": "bag",
}


def build_query(text: str) -> str:
    """中文查询 -> 中英混合查询（长词优先、命中即消耗、同义词去重，离线）。"""
    if not text or not str(text).strip():
        return ""
    raw = str(text).strip()
    remaining = raw
    english_hits = []
    for zh in sorted(ZH2EN, key=len, reverse=True):
        if zh in remaining:
            en = ZH2EN[zh]
            if en not in english_hits:
                english_hits.append(en)
            remaining = remaining.replace(zh, " ")
    if not english_hits:
        return raw
    return f"{raw}, {' '.join(english_hits)}"


# ============================================================================
# 数据源适配层：所有「待匹配照片」都从这里进系统
# ============================================================================
@dataclass
class Candidate:
    """一张待匹配的照片（将来也可以是一条管理员登记记录）。"""
    key: str
    image: Image.Image
    label: str
    source: str = "file"
    path: Optional[Path] = None
    meta: dict = field(default_factory=dict)


def scan_images(directory) -> List[Path]:
    """遍历目录下的图片文件，跳过损坏 / 空文件。"""
    if not directory or not Path(directory).is_dir():
        return []
    found, out, seen = [], [], set()
    for ext in IMAGE_EXTS:
        found.extend(glob.glob(str(Path(directory) / f"*{ext}")))
        found.extend(glob.glob(str(Path(directory) / f"*{ext.upper()}")))
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
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    return Candidate(key=str(path), image=img, label=label or Path(path).stem,
                     source=source, path=Path(path),
                     meta={"mtime": os.path.getmtime(path), "size": os.path.getsize(path)})


def load_candidates_from_dir(directory, source: str = "admin_upload") -> List[Candidate]:
    return [c for p in scan_images(directory) if (c := candidate_from_path(p, source=source))]


def load_candidates_from_uploads(files: Iterable) -> List[Candidate]:
    """Streamlit 的 st.file_uploader 结果 -> Candidate（管理员页面可直接用）。"""
    items = []
    for f in files or []:
        try:
            img = Image.open(f).convert("RGB")
        except Exception:
            continue
        items.append(Candidate(key=getattr(f, "name", str(id(f))), image=img,
                               label=getattr(f, "name", "上传图片"), source="admin_upload", path=None))
    return items


def load_candidates_from_sqlite(db_path, table: str = "lost_item") -> List[Candidate]:
    """管理员登记信息进 SQLite（图片存 BLOB 或路径），表不存在时返回空列表。"""
    import sqlite3
    if not Path(db_path).exists():
        return []
    items = []
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
            items.append(Candidate(key=f"db:{row['id']}", image=img,
                                   label=row["name"] if "name" in cols and row["name"] else f"记录 {row['id']}",
                                   source="db",
                                   meta={"created_at": row["created_at"] if "created_at" in cols else ""}))
        conn.close()
    except Exception:
        return []
    return items


def load_candidates() -> List[Candidate]:
    """当前生效的数据源：所有图库目录 + 管理员目录（存在时自动并入）。"""
    items = []
    for d in IMAGE_DIRS:
        items.extend(load_candidates_from_dir(d, source="file"))
    if ADMIN_UPLOAD_DIR.is_dir():
        items.extend(load_candidates_from_dir(ADMIN_UPLOAD_DIR, source="admin_upload"))
    return items


def scan_library() -> List[Path]:
    """Flask 版用的图库路径列表（去重，含管理员上传目录）。"""
    paths, seen = [], set()
    dirs = list(IMAGE_DIRS) + ([ADMIN_UPLOAD_DIR] if ADMIN_UPLOAD_DIR.is_dir() else [])
    for d in dirs:
        for p in scan_images(d):
            if p not in seen:
                seen.add(p)
                paths.append(p)
    return paths


# ============================================================================
# 图片工具：LOGO 裁白边、缩略图裁剪、PIL -> base64
# ============================================================================
def _logo_b64(path, max_w: int = 560) -> Optional[str]:
    if not path or not os.path.exists(str(path)):
        return None
    try:
        im = Image.open(path).convert("RGB")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        diff = ImageChops.difference(im, bg).convert("L").point(lambda v: 255 if v > 14 else 0)
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


def logo_b64() -> Optional[str]:
    for p in LOGO_PATHS:
        got = _logo_b64(p)
        if got:
            return got
    return None


def _thumb_b64(path: str, size: int = 420) -> Optional[str]:
    """文件路径 -> 居中裁剪的正方形缩略图 base64（来自简洁版 app.py）。"""
    try:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (h - s) // 2 + s, (h - s) // 2 + s))
        im = im.resize((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82)
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


def pil_to_b64(img: Image.Image, quality: int = 85) -> str:
    """Flask API 回传裁片用。"""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def file_to_b64(path: str, max_w: int = 560) -> str:
    """Flask 首页 LOGO 用；读取失败返回空串。"""
    return _logo_b64(path, max_w) or ""


# ============================================================================
# 模型层：Streamlit 用缓存、Flask 用线程锁单例；两者共享同一个构建函数
# ============================================================================
_ENGINE = None
_ENGINE_LOCK = threading.Lock()
_CROP_CACHE: dict = {}          # 同一张图的 YOLO 裁片结果复用


def _build_engine():
    from YOLO import YOLODetector
    from CLIP import CLIPMatcher
    detector = YOLODetector(model_path=str(YOLO_WEIGHTS), conf_threshold=YOLO_CONF)
    matcher = CLIPMatcher(threshold=DEFAULT_THRESHOLD)

    # Mac 上强制 CPU，避免 PyTorch/MPS 崩溃（来自简洁版 app.py）
    if sys.platform == "darwin":
        try:
            if hasattr(detector, "model"):
                detector.model.to("cpu")
            if hasattr(matcher, "device"):
                matcher.device = "cpu"
                matcher.model.to("cpu")
        except Exception:
            pass
    return detector, matcher


def get_engine_flask():
    """Flask 模式：线程锁单例，多次查询秒回。"""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = _build_engine()
        return _ENGINE


if _HAS_STREAMLIT:
    @st.cache_resource(show_spinner=False)
    def get_engine():
        """Streamlit 模式：全局缓存，脚本重跑也不重复加载。"""
        return _build_engine()
else:
    get_engine = get_engine_flask


def crops_for(detector, cand: Candidate) -> List[Image.Image]:
    ck = (cand.key, cand.meta.get("mtime"), cand.meta.get("size"))
    if ck in _CROP_CACHE:
        return _CROP_CACHE[ck]
    target = str(cand.path) if cand.path else cand.image
    crops = detector.detect_and_crop(target) or []
    _CROP_CACHE[ck] = crops
    return crops


# ============================================================================
# 检索逻辑（两界面共用）
# ============================================================================
def mock_search(text: str, paths, k: int):
    """演示模式：固定伪随机分数，不加载模型。"""
    import random
    results = []
    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        w, h = img.size
        seed = int.from_bytes((text + os.path.basename(str(path))).encode("utf-8")[:8], "little")
        rng = random.Random(seed ^ int("smartfind", 36))
        bw, bh = rng.randint(max(1, w // 5), max(2, w // 3)), rng.randint(max(1, h // 5), max(2, h // 3))
        x1 = rng.randint(w // 10, max(w // 10 + 1, w - bw - w // 10))
        y1 = rng.randint(h // 10, max(h // 10 + 1, h - bh - h // 10))
        results.append({"source": str(path), "score": round(rng.uniform(0.35, 0.95), 4),
                        "crop": img.crop((x1, y1, x1 + bw, y1 + bh)), "orig": img})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]


def run_search(query: str, candidates: List[Candidate], *, use_yolo: bool = True,
               top_k: int = DEFAULT_TOPK, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """完整检索：YOLO 抽区域 -> CLIP 批量打分 -> 按图聚合 -> 排序。含中译英增强。"""
    if not candidates:
        return {"q": query, "items": [], "best": 0.0, "n_hit": 0, "n_lib": 0,
                "status": "empty", "msg": "图库为空：请检查 IMAGE_DIRS，或让管理员先上传图片"}
    if MOCK_MODE:
        paths = [c.path for c in candidates if c.path]
        found = mock_search(query, paths, top_k)
        items = [{"key": r["source"], "label": Path(r["source"]).stem, "source": "file",
                  "score": r["score"], "hit": r["score"] >= threshold,
                  "thumb": r["crop"], "full": r["orig"], "n_crops": 1,
                  "detected": True, "debug": [(r["crop"], r["score"])]} for r in found]
        best = items[0]["score"] if items else 0.0
        n_hit = sum(1 for r in items if r["hit"])
        return {"q": query, "items": items, "best": best, "n_hit": n_hit,
                "n_lib": len(candidates), "status": "ok",
                "msg": f"演示模式：命中 {n_hit} 张"}

    detector, matcher = get_engine()
    matcher.threshold = threshold
    q = build_query(query)              # ← 中译英词典在这里生效，两个界面共用

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
                "n_lib": len(candidates), "status": "empty", "msg": "没有可用的照片"}

    res = matcher.match_images(q, pool, top_k=len(pool))
    if res["status"] != "success":
        return {"q": query, "items": [], "best": 0.0, "n_hit": 0,
                "n_lib": len(candidates), "msg": res["message"], "status": "error"}

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
        items.append({"key": key, "label": cand.label, "source": cand.source,
                      "score": best_score, "hit": best_score >= threshold,
                      "thumb": best_crop, "full": cand.image,
                      "n_crops": info[key]["n_crops"], "detected": info[key]["detected"],
                      "debug": hits[:4]})
    items.sort(key=lambda r: r["score"], reverse=True)
    best = items[0]["score"] if items else 0.0
    n_hit = sum(1 for r in items if r["hit"])
    return {"q": query, "items": items[:top_k], "best": best, "n_hit": n_hit,
            "n_lib": len(candidates), "msg": f"命中 {n_hit} 张" if n_hit else "未找到达到阈值的照片",
            "status": "ok"}


# ============================================================================
# 上传：管理员提交图片（两界面共用）
# ============================================================================
def save_upload(filename: str, data_bytes: bytes) -> str:
    """把上传图片保存到管理员目录（不存在则创建），返回落盘路径。"""
    ADMIN_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = ADMIN_UPLOAD_DIR / Path(filename).name
    with open(dest, "wb") as f:
        f.write(data_bytes)
    return str(dest)


# ============================================================================
# 界面一：Streamlit（合并两个 Streamlit 版的全部功能）
# ============================================================================
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


ST_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');
html, body, [data-testid="stAppViewContainer"], .stApp {
    background: radial-gradient(900px 520px at 6% 4%, #ffffff 0%, rgba(255,255,255,0) 62%),
                radial-gradient(820px 480px at 92% 96%, #ffffff 0%, rgba(255,255,255,0) 58%),
                linear-gradient(158deg, #eef8ff 0%, #dcefff 38%, #bfe1fb 100%) !important;
    font-family: "Noto Sans SC", "HarmonyOS Sans SC", "PingFang SC", "Microsoft YaHei UI", sans-serif;
}
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], [data-testid="InputInstructions"] { display: none !important; }
[data-testid="stAppViewContainer"] > .main { background: transparent; }
.stApp { --primary-color: #2f74d0 !important; }
[data-testid="stMainBlockContainer"], .block-container {
    max-width: 700px !important; padding: 24px 28px 30px !important; margin: 128px auto 78px !important;
    background: #ffffff; border-radius: 26px;
    box-shadow: 0 28px 68px rgba(31,86,153,.16), 0 2px 10px rgba(31,86,153,.06);
    position: relative; z-index: 2;
}
[data-testid="stVerticalBlock"] { gap: .45rem; }
.brand { position: fixed; left: 38px; top: 22px; z-index: 9; padding: 10px 14px;
    background: rgba(255,255,255,.97); border: 1.5px solid #d6e8fb; border-radius: 16px;
    box-shadow: 0 10px 26px rgba(31,86,153,.12); transform: rotate(-3deg); }
.brand img { height: 62px; display: block; }
.brand .txt { font-size: 19px; font-weight: 700; color: #1d63b8; letter-spacing: .5px; }
.avatar { position: fixed; right: 42px; top: 30px; z-index: 9; width: 54px; height: 54px;
    border-radius: 50%; background: linear-gradient(180deg, #ffffff, #eaf4ff);
    border: 2px solid #d6e8fb; box-shadow: 0 8px 20px rgba(31,86,153,.13);
    display: flex; align-items: center; justify-content: center; }
[data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; }
.st-key-search [data-testid="stTextInput"] :is([data-baseweb="input"], .react-aria-TextField > div),
.st-key-search [data-testid="stTextInput"] :is([data-testid="stTextInputRootElement"]) {
    position: relative !important; height: 56px !important; display: flex !important; align-items: center !important;
    border-radius: 30px !important; border: 1.6px solid #8ab4e0 !important; background-color: #ffffff !important;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%233f7bc4' stroke-width='1.9' stroke-linecap='round'><circle cx='11' cy='11' r='7'/><line x1='16.4' y1='16.4' x2='21' y2='21'/></svg>");
    background-repeat: no-repeat; background-position: 20px center; background-size: 19px 19px;
    box-shadow: 0 4px 18px rgba(47,116,208,.07) !important; transition: border-color .2s, box-shadow .2s;
}
.st-key-search [data-testid="stTextInput"] :is([data-baseweb="input"], .react-aria-TextField > div):focus-within {
    border-color: #2f74d0 !important; box-shadow: 0 0 0 5px rgba(47,116,208,.10) !important;
}
.st-key-search input { height: 100% !important; border: none !important; outline: none !important;
    background: transparent !important; box-shadow: none !important; padding: 0 22px 0 50px !important;
    font-size: 15px !important; color: #24405e !important; }
.st-key-search input::placeholder { color: #92a8c1 !important; }
.st-key-search [data-testid="stFormSubmitButton"] { position: absolute !important; left: -9999px;
    width: 1px; height: 1px; opacity: 0; }
[data-testid="stTextInput"] label { display: none !important; }
.sec { display: flex; justify-content: space-between; align-items: center; margin: 20px 2px 8px;
    font-size: 13.5px; font-weight: 600; color: #4a6d94; letter-spacing: .4px; }
.sec .count { font-weight: 400; color: #93a9c2; font-size: 12.5px; }
.hist { display: flex; flex-direction: column; gap: 2px; }
.hrow { display: flex; justify-content: space-between; align-items: center; padding: 11px 16px;
    font-size: 14.5px; color: #2c405c; border-radius: 8px; }
.hrow.odd { background: #f1f8ff; }
.hrow .q { max-width: 76%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.hrow .s { font-size: 13px; font-weight: 500; white-space: nowrap; }
.s-ok { color: #17a05e; } .s-run { color: #2f74d0; } .s-no { color: #98a6b8; }
.st-key-hist_toggle button { background: transparent !important; border: none !important; padding: 0 !important;
    color: #6b8bad !important; font-size: 12.5px !important; font-weight: 400 !important;
    height: 20px !important; min-height: 20px !important; line-height: 20px !important; box-shadow: none !important; }
.st-key-hist_toggle button:hover { color: #2f74d0 !important; }
.grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 4px; }
.tile { position: relative; border-radius: 13px; overflow: hidden; aspect-ratio: 1 / 1; background: #e8e8e8;
    display: flex; align-items: center; justify-content: center; color: #b3b3b3; font-size: 12.5px;
    animation: pop .34s cubic-bezier(.2,.8,.3,1) both; }
@keyframes pop { from { opacity: 0; transform: translateY(8px) scale(.97); } to { opacity: 1; transform: none; } }
.tile img { width: 100%; height: 100%; object-fit: cover; display: block; transition: transform .45s ease; }
.tile:hover img { transform: scale(1.07); }
.tile.ref img { filter: saturate(.75) brightness(.97); }
.tile .chip { position: absolute; top: 6px; right: 6px; z-index: 2; font-size: 10.5px; font-weight: 700;
    padding: 2px 7px; border-radius: 999px; background: rgba(47,116,208,.94); color: #fff; }
.tile .chip.ref { background: rgba(255,255,255,.92); color: #7e93a9; }
.tile .cap { position: absolute; left: 0; right: 0; bottom: 0; z-index: 2; padding: 16px 8px 6px;
    font-size: 10.5px; color: #fff; background: linear-gradient(transparent, rgba(0,0,0,.56));
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; opacity: 0; transition: opacity .25s; }
.tile:hover .cap { opacity: 1; }
.tile.empty { background: #fbfdff; border: 1.5px dashed #c5d8ea; color: #b9c9d9; }
.sum { margin-top: 12px; font-size: 12.5px; color: #8ea4bd; display: flex; justify-content: space-between; gap: 10px; }
.sum b { color: #2f74d0; font-weight: 700; }
.pill { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 11.5px; font-weight: 600; }
.pill.ok { background: #e8f8ef; color: #17a05e; }
.pill.no { background: #f2f5f9; color: #8496a8; }
.hint { text-align: center; color: #a3b6c9; font-size: 12.5px; padding: 14px 0 6px; }
.dbgwrap { display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 10px; }
.dbg { position: relative; width: 82px; height: 82px; border-radius: 9px; overflow: hidden; background: #eef3f8; }
.dbg img { width: 100%; height: 100%; object-fit: cover; }
.dbg span { position: absolute; left: 0; right: 0; bottom: 0; text-align: center;
    font-size: 10px; color: #fff; background: rgba(0,0,0,.5); padding: 1px 0; }
[data-testid="stExpander"] { border: 1px solid #eaf2fb !important; border-radius: 14px !important;
    background: #fcfdff; margin-top: 14px; }
[data-testid="stExpander"] details > summary { padding: 9px 14px !important; font-size: 12.5px !important; color: #6b8bad !important; }
[data-testid="stExpander"] details > summary:hover { color: #2f74d0 !important; }
[data-testid="stExpander"] details > summary p { font-size: 12.5px !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { color: #2f74d0 !important; }
[data-testid="stExpander"] [data-testid="stSlider"] { padding-top: 10px; }
[data-testid="stCheckbox"][data-selected="true"] > label > div:first-of-type { background-color: #2f74d0 !important; }
[data-testid="stAlert"] { border-radius: 12px; font-size: 13px; }
</style>
"""


def st_main() -> None:
    """Streamlit 界面：合并了旧 app.py 的 MOCK 开关 + smartfind_app.py 的全部控件。"""
    st.set_page_config(page_title="慧眼寻物 SmartFind", page_icon="🔍", layout="centered")
    st.markdown(ST_CSS, unsafe_allow_html=True)

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

    # 顶部 LOGO + 头像
    lg = logo_b64()
    if lg:
        st.markdown(f'<div class="brand"><img src="data:image/png;base64,{lg}" alt="慧眼寻物 SmartFind"></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="brand"><span class="txt">慧眼寻物 SmartFind</span></div>', unsafe_allow_html=True)
    st.markdown("""<div class="avatar">
      <svg width="29" height="29" viewBox="0 0 24 24" fill="none" stroke="#8b98a8" stroke-width="1.6">
        <circle cx="12" cy="8.2" r="3.6"/><path d="M4.6 20c1.2-3.9 4-5.8 7.4-5.8s6.2 1.9 7.4 5.8"/>
      </svg></div>""", unsafe_allow_html=True)

    if MOCK_MODE:
        st.info("演示模式（MOCK_MODE）：不加载真实模型，秒出示例结果。去掉 --mock / SMARTFIND_MOCK 即为真实识别。")

    # 搜索栏（回车即搜）
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

    # 历史记录
    head_l, head_r = st.columns([4, 1], vertical_alignment="center")
    with head_l:
        st.markdown('<div class="sec" style="margin:18px 2px 4px">历史记录</div>', unsafe_allow_html=True)
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
                rows.append(f'<div class="hrow{" odd" if i % 2 == 0 else ""}">'
                            f'<span class="q">{html.escape(h["q"])}</span>'
                            f'<span class="s {cls}">{h["status"]}</span></div>')
            st.markdown(f'<div class="hist">{"".join(rows)}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="hist"><div class="hrow odd">'
                        '<span class="q" style="color:#a8b6c6">暂无搜索记录</span>'
                        '<span class="s s-no">—</span></div></div>', unsafe_allow_html=True)

    # 执行查询
    if ss.pending:
        pending = ss.pending
        ss.pending = None
        with st.spinner("正在比对照片区…"):
            try:
                result = run_search(pending, candidates, use_yolo=ss.use_yolo,
                                    top_k=ss.top_k, threshold=ss.threshold)
            except Exception as exc:
                result = {"q": pending, "items": [], "best": 0.0, "n_hit": 0,
                          "n_lib": len(candidates), "status": "error", "msg": f"检索失败：{exc}"}
        ss.result = result
        for h in ss.history:
            if h["q"] == pending and h["status"] == "查询中":
                h["status"] = "已找到" if result["n_hit"] else (
                    "出错" if result["status"] == "error" else "未找到")
                h["score"] = result["best"]
                break
        st.rerun()

    result = ss.result
    # 结果区
    if result and result["status"] in ("error", "empty") and not result["items"]:
        (st.error if result["status"] == "error" else st.info)(result["msg"])
        st.markdown(f'<div class="sec">备选图片<span class="count">图库 {result["n_lib"]} 张</span></div>',
                    unsafe_allow_html=True)
        st.markdown(_grid_html([]), unsafe_allow_html=True)
    elif result and result["items"]:
        items = result["items"]
        for it in items:
            it["hit"] = it["score"] >= ss.threshold
        n_hit = sum(1 for it in items if it["hit"])
        st.markdown(f'<div class="sec">备选图片<span class="count">按匹配度排序 · 图库 {result["n_lib"]} 张</span></div>',
                    unsafe_allow_html=True)
        st.markdown(_grid_html(items, ss.top_k, ss.show_full), unsafe_allow_html=True)
        pill = '<span class="pill ok">已找到</span>' if n_hit else '<span class="pill no">未达阈值</span>'
        extra = "（灰色为相似度偏低的参考画面）" if n_hit < len(items) else ""
        st.markdown(f'<div class="sum"><span>{pill} 「{html.escape(result["q"])}」命中 '
                    f'<b>{n_hit}</b> 张 {extra}</span>'
                    f'<span>最高相似度 <b>{result["best"] * 100:.1f}%</b></span></div>', unsafe_allow_html=True)
        if ss.show_debug:
            with st.expander("调试视图：YOLO 裁片与 CLIP 打分", expanded=True):
                st.caption("每张小图是 YOLO 从原照片里裁出的候选区域，下方数字为与输入描述的 CLIP 相似度（×100）。")
                st.markdown(_debug_html(items[:4]), unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="sec">备选图片<span class="count">图库 {len(candidates)} 张</span></div>',
                    unsafe_allow_html=True)
        st.markdown(_grid_html([]), unsafe_allow_html=True)
        st.markdown('<div class="hint">输入物品名称或特征开始查找，例如「红色双肩包」「黄色保温杯」</div>',
                    unsafe_allow_html=True)

    # 检索设置（含 MOCK 开关 + 旧 app.py 的全部设置项）
    lib_names = "、".join(d.name for d in IMAGE_DIRS)
    with st.expander(f"检索设置 · 图库 {len(candidates)} 张（{lib_names}）", expanded=False):
        st.caption("YOLO 裁出候选区域，CLIP 计算相似度；关掉 YOLO 可对比「整图直接比对」。"
                   "中文查询会自动追加英文提示以增强 CLIP 匹配。")
        ss.top_k = int(st.slider("返回图片数量", 4, 16, int(ss.top_k), 4))
        ss.threshold = float(st.slider("相似度阈值（低于该值视为未命中）", 0.05, 0.60, float(ss.threshold), 0.01))
        ss.use_yolo = st.toggle("启用 YOLO 裁片", value=ss.use_yolo,
                                help="关闭后整张照片直接和文字比对（改完需重新搜索）")
        ss.show_full = st.toggle("图块显示原始照片", value=ss.show_full,
                                 help="关闭则显示 YOLO 匹配到的裁片")
        ss.show_debug = st.toggle("显示调试视图", value=ss.show_debug,
                                  help="展开显示每张照片的候选裁片与得分")
        b1, b2 = st.columns(2)
        if b1.button("重新扫描图库", use_container_width=True):
            _CROP_CACHE.clear()
            st.rerun()
        if b2.button("清空历史", use_container_width=True):
            ss.history, ss.result = [], None
            st.rerun()
        # 管理员上传（Streamlit 侧入口，落到与 Flask 同一目录）
        with st.expander("管理员：上传图片到图库", expanded=False):
            up = st.file_uploader("选择图片（可多选）", type=["jpg", "jpeg", "png", "bmp", "webp"],
                                  accept_multiple_files=True)
            if st.button("上传入库") and up:
                saved = [save_upload(f.name, f.getbuffer()) for f in up]
                _CROP_CACHE.clear()
                st.success(f"已入库 {len(saved)} 张到 {ADMIN_UPLOAD_DIR}")



# ============================================================================
# 界面二：Flask 独立网页（保留 smartfind_web.py 的完整 HTML / 路由 / 管理员上传）
# ============================================================================
def flask_real_search(query: str, k: int):
    """Flask 正式流程：中译英 -> 每张图 YOLO 切图 -> CLIP 打分 -> 全局 Top-K。"""
    detector, matcher = get_engine_flask()
    q = build_query(query)
    results = []
    for path in scan_library():
        try:
            crops = detector.detect_and_crop(str(path))
            if not crops:
                continue
            res = matcher.match_images(q, crops, top_k=3)
            if res.get("status") != "success":
                continue
            for crop, score in res["top_k_results"]:
                results.append({"score": round(float(score), 4),
                                "source": Path(path).name, "crop_b64": pil_to_b64(crop)})
        except Exception as e:
            print(f"[搜索] 跳过 {path}: {e}")
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]


def flask_mock_search(query: str, k: int):
    results = []
    for path in scan_library():
        seed = int.from_bytes((query + Path(path).name).encode("utf-8")[:8], "little")
        import random
        score = round(random.Random(seed).uniform(0.30, 0.95), 4)
        results.append({"score": score, "source": Path(path).name, "crop_b64": ""})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]


PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>慧眼寻物 SmartFind</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body {
    min-height: 100vh;
    background: linear-gradient(180deg, #e8f3fc 0%, #cfe6f8 45%, #a9d3f2 100%);
    font-family: "Noto Sans SC", "PingFang SC", "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
    overflow-x: hidden;
}
.brand-wrap {
    position: fixed; left: 42px; top: 26px; z-index: 9;
    padding: 10px 18px; background: rgba(255,255,255,.96);
    border: 1.5px solid #d3e6fb; border-radius: 18px;
    box-shadow: 0 10px 26px rgba(31,86,153,.13); transform: rotate(-3deg);
}
.brand-wrap img { height: 66px; display: block; }
.avatar-wrap {
    position: fixed; right: 48px; top: 34px; z-index: 9;
    width: 54px; height: 54px; border-radius: 50%;
    background: linear-gradient(180deg, #ffffff, #eaf4ff);
    border: 2px solid #d3e6fb; box-shadow: 0 8px 20px rgba(31,86,153,.14);
    display: flex; align-items: center; justify-content: center;
}
.card {
    max-width: 680px; margin: 130px auto 70px; padding: 28px 34px 34px;
    background: #ffffff; border-radius: 26px; position: relative; z-index: 2;
    box-shadow: 0 26px 64px rgba(31,86,153,.18), 0 2px 10px rgba(31,86,153,.06);
}
.search-box { position: relative; }
.search-box input {
    width: 100%; height: 58px; border-radius: 30px;
    border: 1.6px solid #7aa9dd; background: #fff; outline: none;
    padding: 0 66px 0 54px; font-size: 15px; color: #24405e;
    box-shadow: 0 0 0 5px rgba(255,255,255,.75);
    transition: border-color .2s, box-shadow .2s;
}
.search-box input::placeholder { color: #93a9c2; }
.search-box input:focus { border-color: #2f74d0; box-shadow: 0 0 0 6px rgba(47,116,208,.12); }
.search-box .icon {
    position: absolute; left: 22px; top: 50%; transform: translateY(-50%);
    width: 19px; height: 19px; pointer-events: none;
}
.search-box .go {
    position: absolute; right: 8px; top: 8px; height: 42px; padding: 0 20px;
    border: none; border-radius: 22px; cursor: pointer;
    background: #2f74d0; color: #fff; font-size: 14px; font-weight: 600;
    transition: background .2s;
}
.search-box .go:hover { background: #2565b8; }
.search-box .go:disabled { background: #9dbfe6; cursor: default; }
.sec {
    display: flex; justify-content: space-between; align-items: center;
    margin: 20px 0 6px; font-size: 14px; font-weight: 600; color: #4a6d94;
}
.sec .fold { font-size: 13px; font-weight: 400; color: #6b8bad; cursor: pointer; user-select: none; }
.sec .fold:hover { color: #2f74d0; }
.hist { display: flex; flex-direction: column; gap: 2px; }
.hrow {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 18px; font-size: 14.5px; color: #2c405c; border-radius: 8px;
    cursor: pointer;
}
.hrow:hover { background: #e3f1fd; }
.hrow.odd { background: #f0f8ff; }
.hrow .q { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 74%; }
.hrow .s { font-size: 13.5px; font-weight: 500; white-space: nowrap; }
.s-ok  { color: #17a05e; }
.s-run { color: #2f74d0; }
.s-no  { color: #98a6b8; }
.grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 6px; }
.tile {
    position: relative; border-radius: 13px; overflow: hidden;
    aspect-ratio: 4 / 3; background: #e8e8e8;
    display: flex; align-items: center; justify-content: center;
    color: #b0b0b0; font-size: 12.5px;
}
.tile img { width: 100%; height: 100%; object-fit: cover; display: block; transition: transform .4s; }
.tile:hover img { transform: scale(1.09); }
.tile.empty { background: #f6fafd; border: 2px dashed #c3d7ec; color: #b6c8db; }
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
.tile .loading {
    width: 22px; height: 22px; border: 3px solid #cfe0f2;
    border-top-color: #2f74d0; border-radius: 50%;
    animation: spin .8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.sum { font-size: 12.5px; color: #8ea4bd; margin-top: 10px; text-align: right; min-height: 16px; }
.sum b { color: #2f74d0; font-weight: 700; }
.banner {
    display: none; margin: 12px 0 4px; padding: 10px 16px; border-radius: 12px;
    font-size: 13.5px; font-weight: 600;
}
.banner.ok  { display: block; background: #e7f7ee; color: #17a05e; }
.banner.no  { display: block; background: #f2f5f9; color: #98a6b8; }
.fab {
    position: fixed; right: 26px; bottom: 26px; z-index: 20;
    width: 48px; height: 48px; border-radius: 50%; border: 1.5px solid #d7e6f7;
    background: #fff; color: #4a7cb8; font-size: 18px; cursor: pointer;
    box-shadow: 0 8px 22px rgba(31,86,153,.16);
}
.fab:hover { border-color: #7aa9dd; color: #2f74d0; }
.admin-panel {
    position: fixed; right: 26px; bottom: 86px; z-index: 20; width: 300px;
    background: #fff; border-radius: 16px; padding: 18px;
    box-shadow: 0 18px 40px rgba(31,86,153,.18); display: none;
}
.admin-panel.show { display: block; }
.admin-panel h4 { font-size: 14px; color: #2c405c; margin-bottom: 8px; }
.admin-panel p { font-size: 12px; color: #8ea4bd; margin-bottom: 10px; line-height: 1.6; }
.admin-panel input[type=file] { font-size: 12px; margin-bottom: 10px; width: 100%; }
.admin-panel button {
    width: 100%; height: 36px; border: none; border-radius: 10px; cursor: pointer;
    background: #2f74d0; color: #fff; font-size: 13px; font-weight: 600;
}
.admin-msg { font-size: 12px; margin-top: 8px; color: #17a05e; display: none; }
</style>
</head>
<body>
<div class="brand-wrap">
    <img src="data:image/png;base64,__LOGO_B64__" alt="慧眼寻物 SmartFind"
         onerror="this.parentNode.innerHTML='<span style=&quot;font-size:20px;font-weight:700;color:#1d63b8&quot;>慧眼寻物 SmartFind</span>'">
</div>
<div class="avatar-wrap">
    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#8b98a8" stroke-width="1.6">
        <circle cx="12" cy="8.2" r="3.6"/>
        <path d="M4.6 20c1.2-3.9 4-5.8 7.4-5.8s6.2 1.9 7.4 5.8"/>
    </svg>
</div>
<div class="card">
    <div class="search-box">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="#3f7bc4" stroke-width="2" stroke-linecap="round">
            <circle cx="11" cy="11" r="7"/><line x1="16.5" y1="16.5" x2="21" y2="21"/>
        </svg>
        <input id="q" type="text" placeholder="输入名称、特征，搜索失物" autocomplete="off">
        <button class="go" id="go">搜索</button>
    </div>
    <div class="sec"><span>历史记录</span><span class="fold" id="fold">折叠</span></div>
    <div class="hist" id="hist"></div>
    <div class="banner" id="banner"></div>
    <div class="sec" style="margin-top:22px"><span>匹配结果</span><span id="libinfo"></span></div>
    <div class="grid" id="grid"></div>
    <div class="sum" id="sum"></div>
</div>
<button class="fab" id="fab" title="管理员上传图片">⚙</button>
<div class="admin-panel" id="apanel">
    <h4>管理员 · 提交图片</h4>
    <p>图片将保存到图库目录 <code>web/admin_uploads/</code>，保存后即可被用户搜索匹配。</p>
    <input type="file" id="afile" accept=".jpg,.jpeg,.png,.bmp,.webp">
    <button id="aup">上传入库</button>
    <div class="admin-msg" id="amsg"></div>
</div>
<script>
const SLOTS = 8;
const grid = document.getElementById('grid');
const histEl = document.getElementById('hist');
const sumEl = document.getElementById('sum');
const banner = document.getElementById('banner');
const libinfo = document.getElementById('libinfo');
let history = JSON.parse(localStorage.getItem('sf_history') || '[]');
let histOpen = true;
function renderHist() {
    if (!histOpen) { histEl.innerHTML = ''; return; }
    if (!history.length) {
        histEl.innerHTML = '<div class="hrow odd"><span class="q" style="color:#a8b6c6">暂无搜索记录</span><span class="s s-no">—</span></div>';
        return;
    }
    histEl.innerHTML = history.map((h, i) => {
        const cls = h.status === '已找到' ? 's-ok' : (h.status === '查询中' ? 's-run' : 's-no');
        return `<div class="hrow ${i % 2 === 0 ? 'odd' : ''}" data-q="${h.q}">
            <span class="q">${h.q}</span><span class="s ${cls}">${h.status}</span></div>`;
    }).join('');
    histEl.querySelectorAll('.hrow').forEach(r => {
        r.onclick = () => { document.getElementById('q').value = r.dataset.q; doSearch(); };
    });
}
document.getElementById('fold').onclick = e => {
    histOpen = !histOpen;
    e.target.textContent = histOpen ? '折叠' : '展开';
    renderHist();
};
function renderGrid(items) {
    let html = '';
    const list = items || [];
    for (let i = 0; i < SLOTS; i++) {
        if (list[i]) {
            const r = list[i];
            const body = r.crop_b64
                ? `<img src="data:image/jpeg;base64,${r.crop_b64}" alt="">`
                : '备选图片';
            html += `<div class="tile" title="匹配度 ${(r.score * 100).toFixed(1)}%">
                ${body}
                <span class="chip">${Math.round(r.score * 100)}%</span>
                <span class="fn">${r.source}</span></div>`;
        } else if (items === 'loading') {
            html += '<div class="tile empty"><div class="loading"></div></div>';
        } else {
            html += '<div class="tile empty">备选图片</div>';
        }
    }
    grid.innerHTML = html;
}
async function doSearch() {
    const input = document.getElementById('q');
    const goBtn = document.getElementById('go');
    const query = input.value.trim();
    if (!query) { input.focus(); return; }
    history = history.filter(h => h.q !== query);
    history.unshift({ q: query, status: '查询中' });
    history = history.slice(0, 6);
    localStorage.setItem('sf_history', JSON.stringify(history));
    renderHist();
    goBtn.disabled = true;
    renderGrid('loading');
    sumEl.textContent = '';
    banner.className = 'banner';
    libinfo.textContent = '';
    try {
        const resp = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });
        const data = await resp.json();
        const st = data.is_found ? '已找到' : '未找到';
        history = history.map(h => h.q === query ? { ...h, status: st } : h);
        localStorage.setItem('sf_history', JSON.stringify(history));
        renderHist();
        if (data.status === 'error') {
            banner.className = 'banner no';
            banner.textContent = data.message;
            renderGrid(null);
            return;
        }
        renderGrid(data.results);
        libinfo.textContent = `图库 ${data.library_size} 张`;
        banner.className = 'banner ' + (data.is_found ? 'ok' : 'no');
        banner.textContent = data.is_found
            ? `✅ 找到疑似目标！最高匹配度 ${(data.best_score * 100).toFixed(1)}%`
            : `❌ 未找到目标（最高匹配度 ${(data.best_score * 100).toFixed(1)}%，未达阈值）`;
        sumEl.innerHTML = `「${query}」共 ${data.results.length} 张候选画面 · 已按匹配度排序 · 最高 <b>${(data.best_score * 100).toFixed(1)}%</b>`;
    } catch (err) {
        banner.className = 'banner no';
        banner.textContent = '网络错误，请确认服务已启动';
        renderGrid(null);
    } finally {
        goBtn.disabled = false;
    }
}
document.getElementById('go').onclick = doSearch;
document.getElementById('q').addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });
const apanel = document.getElementById('apanel');
document.getElementById('fab').onclick = () => apanel.classList.toggle('show');
document.getElementById('aup').onclick = async () => {
    const f = document.getElementById('afile').files[0];
    const msg = document.getElementById('amsg');
    if (!f) { msg.style.display = 'block'; msg.style.color = '#d05a2f'; msg.textContent = '请先选择图片'; return; }
    const fd = new FormData();
    fd.append('file', f);
    const resp = await fetch('/admin/upload', { method: 'POST', body: fd });
    const data = await resp.json();
    msg.style.display = 'block';
    msg.style.color = data.status === 'success' ? '#17a05e' : '#d05a2f';
    msg.textContent = data.message;
};
renderHist();
renderGrid(null);
</script>
</body>
</html>
"""


def flask_main() -> None:
    """Flask 独立网页：保留全部路由 + 管理员上传 + 中译英查询。"""
    from flask import Flask, request, jsonify, Response
    app = Flask(__name__)

    @app.route("/api/search", methods=["POST"])
    def api_search():
        data = request.get_json(silent=True) or {}
        query = str(data.get("query", "")).strip()
        if not query:
            return jsonify({"status": "error", "message": "输入为空，请重新输入物品描述"})
        items = flask_mock_search(query, GRID_SLOTS) if MOCK_MODE else flask_real_search(query, GRID_SLOTS)
        best = items[0]["score"] if items else 0.0
        return jsonify({"status": "success", "is_found": best >= DEFAULT_THRESHOLD,
                        "best_score": best, "library_size": len(scan_library()), "results": items})

    @app.route("/admin/upload", methods=["POST"])
    def admin_upload():
        f = request.files.get("file")
        if f is None or not f.filename:
            return jsonify({"status": "error", "message": "未收到图片文件"}), 400
        if not re.search(r"\.(jpe?g|png|bmp|webp)$", f.filename, re.I):
            return jsonify({"status": "error", "message": "仅支持 jpg/png/bmp/webp"}), 400
        dest = save_upload(f.filename, f.read())
        return jsonify({"status": "success", "message": f"已入库: {dest}"})

    @app.route("/")
    def index():
        html = PAGE_HTML.replace("__LOGO_B64__", file_to_b64(str(LOGO_PATHS[0])))
        return Response(html, mimetype="text/html")

    print("=" * 52)
    print("  慧眼寻物 SmartFind 正在启动（Flask 独立网页）...")
    print(f"  模式: {'快速演示(不加载模型)' if MOCK_MODE else '真实 YOLO + CLIP'}")
    print(f"  图库: {len(scan_library())} 张照片")
    print(f"  浏览器打开  http://{API_HOST}:{API_PORT}")
    print("=" * 52)
    app.run(host=API_HOST, port=API_PORT, debug=False)



# ============================================================================
# 入口：根据启动方式自动选择界面
# ============================================================================
def _running_in_streamlit() -> bool:
    """streamlit run 时为真；直接 python 执行时为假。"""
    if not _HAS_STREAMLIT:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    argv = [a.lower() for a in sys.argv[1:] if not a.startswith("--")]
    if "flask" in argv:
        flask_main()
    elif "st" in argv or "streamlit" in argv:
        st_main()
    elif _running_in_streamlit():
        st_main()
    else:
        # 直接 python app.py：默认起 Flask 独立网页
        flask_main()
