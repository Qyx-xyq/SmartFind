# ============================================================
# 慧眼寻物 SmartFind —— 前端交互网页（正式版）
# ============================================================
# 运行方式：在项目根目录 D:/SmartFind 下执行
#     python web/smartfind_web.py
# 然后浏览器打开  http://127.0.0.1:5000
#
# 【与旧版 web/app.py 的关系】
# 旧版 Streamlit 页面是测试版，本文件是完全独立的新版页面，
# 互不影响。不想用旧版直接删掉 app.py 也不影响本文件。
#
# 【核心流程】
# 用户输入物品描述 -> 遍历图库 -> YOLO 切出候选子图(秦宇霄)
# -> CLIP 图文匹配打分(黄涵) -> 按匹配度从高到低弹出图片
# -> 图片不够的位置用虚线矩形占位
#
# 【管理员图片接入（将来只需两步）】
# 1. 管理员页面做好后，把图片 POST 到  /admin/upload  即可，
#    例如:  curl -F "file=@新照片.jpg" http://127.0.0.1:5000/admin/upload
#    上传的图片会自动保存进图库目录，刷新页面即可参与匹配；
# 2. 如果管理员希望使用单独的文件夹，只需把下方 ADMIN_UPLOAD_DIR
#    和 IMAGE_DIRS 改成那个文件夹路径，其余代码不用动。
#
# 【当前验证数据】
# 图库 = web/test_images/ 下的 9 张照片 + data/ 下的示例照片，
# 均为项目里已有的图片，用于在管理员系统就绪前做功能验证。
# ============================================================

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 避免 OpenMP 冲突（与 CLIP.py 保持一致）

import io
import re
import base64
import threading

from flask import Flask, request, jsonify, Response

# —— 接入队友的核心模块（接口与他们的现有代码完全一致）——
from YOLO import YOLODetector      # detect_and_crop(图片路径) -> [PIL 子图]
from CLIP import CLIPMatcher       # match_images(文本, [PIL 子图], top_k) -> dict

# ============================================================================
# ★★★ 配置区（将来接入管理员系统时只需要改这里）★★★
# ============================================================================
MOCK_MODE         = False                  # True=跳过模型快速演示；False=真实 YOLO+CLIP
IMAGE_DIRS        = ["web/test_images", "data"]   # 图库目录（用项目已有照片验证）
ADMIN_UPLOAD_DIR  = "web/test_images"      # 管理员上传的图片保存目录
LOGO_PATH         = os.path.join("web", "logo.png")   # 项目 LOGO（已经就是设计稿那张）
TOP_K             = 8                      # 结果网格固定 8 个格子，图片不足用虚线框占位
THRESHOLD         = 0.25                   # 相似度判定阈值（与 CLIP.py 默认一致）

# CLIP 官方模型对中文支持较弱，这里内置一张"常见失物词 -> 英文"对照表。
# 用户输入包含这些词时，自动把英文提示追加给 CLIP，提高匹配精度。
# 将来若换成中文 CLIP 模型（如 Chinese-CLIP），删掉这个字典即可。
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
    """把中文查询转成 CLIP 更容易理解的英文/中英混合查询。

    离线词典加强版：
    - 长词优先：按词条长度降序匹配，避免"双肩包"被"包"之类短词拆散
    - 命中即消耗：匹配到的词从待匹配文本里移除，避免同一段文字被重复翻译
    - 同义词归一：多个中文说法映射到同一个英文，自动去重
    - 中英拼接：保留用户原文 + 追加英文提示，CLIP 中英文都能读到
    全程离线，无网络请求，展示现场断网也能正常工作。
    """
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
# 资源层：LOGO 与图片转 base64（前端才能直接显示本地图片）
# ============================================================================
def file_to_b64(path: str, max_w: int = 560) -> str:
    """读取图片（LOGO 自动裁掉白边）转 base64，读取失败返回空串。"""
    if not path or not os.path.exists(path):
        return ""
    try:
        from PIL import Image, ImageChops
        im = Image.open(path).convert("RGB")
        if path.endswith("logo.png"):          # LOGO 裁白边
            bg = Image.new("RGB", im.size, (255, 255, 255))
            diff = ImageChops.difference(im, bg).convert("L").point(
                lambda v: 255 if v > 14 else 0)
            bbox = diff.getbbox()
            if bbox:
                im = im.crop(bbox)
        if im.width > max_w:
            im = im.resize((max_w, round(im.height * max_w / im.width)))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def pil_to_b64(img, quality=85) -> str:
    """PIL 子图 -> base64 JPEG（直接塞进 JSON 返回给前端）。"""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def scan_library():
    """扫描所有图库目录，返回图片绝对路径列表（去重）。"""
    paths, seen = [], set()
    for d in IMAGE_DIRS:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if re.search(r"\.(jpe?g|png|bmp|webp)$", name, re.I):
                p = os.path.abspath(os.path.join(d, name))
                if p not in seen:
                    seen.add(p)
                    paths.append(p)
    return paths


# ============================================================================
# 模型层：全局只加载一次（单例 + 线程锁，多次查询秒回）
# ============================================================================
_lock = threading.Lock()
_detector, _matcher = None, None


def load_models():
    global _detector, _matcher
    with _lock:
        if _detector is None:
            _detector = YOLODetector(model_path="models/yolov8n.pt", conf_threshold=0.15)
            _matcher = CLIPMatcher(threshold=THRESHOLD)
    return _detector, _matcher


def mock_search(query: str, k: int):
    """演示模式：不加载模型，用固定伪随机分数，方便快速看页面效果。"""
    import random
    results = []
    for path in scan_library():
        seed = int.from_bytes((query + os.path.basename(path)).encode("utf-8")[:8], "little")
        score = round(random.Random(seed).uniform(0.30, 0.95), 4)
        results.append({"score": score, "source": os.path.basename(path),
                        "crop_b64": ""})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]


def real_search(query: str, k: int):
    """正式流程：每张图库照片 YOLO 切图 -> CLIP 打分 -> 全局排序取 Top-K。"""
    detector, matcher = load_models()
    q = build_query(query)
    results = []
    for path in scan_library():
        try:
            crops = detector.detect_and_crop(path)
            if not crops:
                continue
            res = matcher.match_images(q, crops, top_k=3)
            if res.get("status") != "success":
                continue
            for crop, score in res["top_k_results"]:
                results.append({
                    "score": round(float(score), 4),
                    "source": os.path.basename(path),
                    "crop_b64": pil_to_b64(crop),   # 只回传 base64，避免 PIL 对象无法 JSON 序列化
                })
        except Exception as e:
            print(f"[搜索] 跳过 {path}: {e}")
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]


# ============================================================================
# 后端路由
# ============================================================================
app = Flask(__name__)


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()
    if not query:
        return jsonify({"status": "error", "message": "输入为空，请重新输入物品描述"})

    items = mock_search(query, TOP_K) if MOCK_MODE else real_search(query, TOP_K)
    best = items[0]["score"] if items else 0.0
    return jsonify({
        "status": "success",
        "is_found": best >= THRESHOLD,
        "best_score": best,
        "library_size": len(scan_library()),
        "results": items,
    })


@app.route("/admin/upload", methods=["POST"])
def admin_upload():
    """管理员图片提交接口（管理员页面就绪后直接 POST 图片文件到这里）。"""
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"status": "error", "message": "未收到图片文件"}), 400
    if not re.search(r"\.(jpe?g|png|bmp|webp)$", f.filename, re.I):
        return jsonify({"status": "error", "message": "仅支持 jpg/png/bmp/webp"}), 400
    os.makedirs(ADMIN_UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(ADMIN_UPLOAD_DIR, f.filename)
    f.save(save_path)
    return jsonify({"status": "success", "message": f"已入库: {save_path}"})


@app.route("/")
def index():
    logo_b64 = file_to_b64(LOGO_PATH)
    html = PAGE_HTML.replace("__LOGO_B64__", logo_b64)
    return Response(html, mimetype="text/html")


# ============================================================================
# 前端页面（排版按设计稿：顶部搜索栏 -> 历史记录 -> 匹配图片网格）
# ============================================================================
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

/* ---------- 左上 LOGO 徽标（贴纸效果） ---------- */
.brand-wrap {
    position: fixed; left: 42px; top: 26px; z-index: 9;
    padding: 10px 18px; background: rgba(255,255,255,.96);
    border: 1.5px solid #d3e6fb; border-radius: 18px;
    box-shadow: 0 10px 26px rgba(31,86,153,.13); transform: rotate(-3deg);
}
.brand-wrap img { height: 66px; display: block; }

/* ---------- 右上头像 ---------- */
.avatar-wrap {
    position: fixed; right: 48px; top: 34px; z-index: 9;
    width: 54px; height: 54px; border-radius: 50%;
    background: linear-gradient(180deg, #ffffff, #eaf4ff);
    border: 2px solid #d3e6fb; box-shadow: 0 8px 20px rgba(31,86,153,.14);
    display: flex; align-items: center; justify-content: center;
}

/* ---------- 中央白色大卡片 ---------- */
.card {
    max-width: 680px; margin: 130px auto 70px; padding: 28px 34px 34px;
    background: #ffffff; border-radius: 26px; position: relative; z-index: 2;
    box-shadow: 0 26px 64px rgba(31,86,153,.18), 0 2px 10px rgba(31,86,153,.06);
}

/* ---------- 搜索胶囊 ---------- */
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

/* ---------- 历史记录 ---------- */
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

/* ---------- 匹配结果网格 ---------- */
.grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 6px; }
.tile {
    position: relative; border-radius: 13px; overflow: hidden;
    aspect-ratio: 4 / 3; background: #e8e8e8;
    display: flex; align-items: center; justify-content: center;
    color: #b0b0b0; font-size: 12.5px;
}
.tile img { width: 100%; height: 100%; object-fit: cover; display: block; transition: transform .4s; }
.tile:hover img { transform: scale(1.09); }
.tile.empty {
    background: #f6fafd; border: 2px dashed #c3d7ec; color: #b6c8db;
}
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

/* ---------- 结果小结 ---------- */
.sum { font-size: 12.5px; color: #8ea4bd; margin-top: 10px; text-align: right; min-height: 16px; }
.sum b { color: #2f74d0; font-weight: 700; }
.banner {
    display: none; margin: 12px 0 4px; padding: 10px 16px; border-radius: 12px;
    font-size: 13.5px; font-weight: 600;
}
.banner.ok  { display: block; background: #e7f7ee; color: #17a05e; }
.banner.no  { display: block; background: #f2f5f9; color: #98a6b8; }

/* ---------- 右下角管理员入口（管理员页面上线后可直接移除） ---------- */
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

<!-- LOGO（项目设计稿原版） -->
<div class="brand-wrap">
    <img src="data:image/png;base64,__LOGO_B64__" alt="慧眼寻物 SmartFind"
         onerror="this.parentNode.innerHTML='<span style=&quot;font-size:20px;font-weight:700;color:#1d63b8&quot;>慧眼寻物 SmartFind</span>'">
</div>

<!-- 右上头像 -->
<div class="avatar-wrap">
    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#8b98a8" stroke-width="1.6">
        <circle cx="12" cy="8.2" r="3.6"/>
        <path d="M4.6 20c1.2-3.9 4-5.8 7.4-5.8s6.2 1.9 7.4 5.8"/>
    </svg>
</div>

<div class="card">
    <!-- ① 顶部搜索栏 -->
    <div class="search-box">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="#3f7bc4" stroke-width="2" stroke-linecap="round">
            <circle cx="11" cy="11" r="7"/><line x1="16.5" y1="16.5" x2="21" y2="21"/>
        </svg>
        <input id="q" type="text" placeholder="输入名称、特征，搜索失物" autocomplete="off">
        <button class="go" id="go">搜索</button>
    </div>

    <!-- ② 历史记录 -->
    <div class="sec">
        <span>历史记录</span><span class="fold" id="fold">折叠</span>
    </div>
    <div class="hist" id="hist"></div>

    <!-- ③ 匹配结果提示 -->
    <div class="banner" id="banner"></div>

    <!-- ④ 匹配图片网格（8 格，不足虚线占位） -->
    <div class="sec" style="margin-top:22px"><span>匹配结果</span><span id="libinfo"></span></div>
    <div class="grid" id="grid"></div>
    <div class="sum" id="sum"></div>
</div>

<!-- 管理员临时上传入口（验证用，管理员页面上线后可整体删除） -->
<button class="fab" id="fab" title="管理员上传图片">⚙</button>
<div class="admin-panel" id="apanel">
    <h4>管理员 · 提交图片</h4>
    <p>图片将保存到图库目录 <code>web/test_images/</code>，保存后即可被用户搜索匹配。</p>
    <input type="file" id="afile" accept=".jpg,.jpeg,.png,.bmp,.webp">
    <button id="aup">上传入库</button>
    <div class="admin-msg" id="amsg"></div>
</div>

<script>
// ================= 状态 =================
const SLOTS = 8;                 // 网格固定 8 格
const grid = document.getElementById('grid');
const histEl = document.getElementById('hist');
const sumEl = document.getElementById('sum');
const banner = document.getElementById('banner');
const libinfo = document.getElementById('libinfo');
let history = JSON.parse(localStorage.getItem('sf_history') || '[]');
let histOpen = true;

// ================= 历史记录 =================
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
    // 点击历史记录直接再次搜索
    histEl.querySelectorAll('.hrow').forEach(r => {
        r.onclick = () => { document.getElementById('q').value = r.dataset.q; doSearch(); };
    });
}
document.getElementById('fold').onclick = e => {
    histOpen = !histOpen;
    e.target.textContent = histOpen ? '折叠' : '展开';
    renderHist();
};

// ================= 结果网格 =================
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
            html += '<div class="tile empty">备选图片</div>';   // 虚线占位框
        }
    }
    grid.innerHTML = html;
}

// ================= 搜索 =================
async function doSearch() {
    const input = document.getElementById('q');
    const goBtn = document.getElementById('go');
    const query = input.value.trim();
    if (!query) { input.focus(); return; }

    // 1. 历史记录置顶并标记"查询中"
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

        // 2. 更新历史状态
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

        // 3. 按匹配度从高到低展示（后端已排好序）
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

// ================= 管理员上传（临时入口） =================
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

// ================= 初始渲染 =================
renderHist();
renderGrid(null);
</script>
</body>
</html>
"""

# ============================================================================
# 启动
# ============================================================================
if __name__ == "__main__":
    print("=" * 52)
    print("  慧眼寻物 SmartFind 正在启动...")
    print(f"  模式: {'快速演示(不加载模型)' if MOCK_MODE else '真实 YOLO + CLIP'}")
    print(f"  图库: {scan_library().__len__()} 张照片  ->  {IMAGE_DIRS}")
    print("  浏览器打开  http://127.0.0.1:5000")
    print("=" * 52)
    app.run(host="127.0.0.1", port=5000, debug=False)
