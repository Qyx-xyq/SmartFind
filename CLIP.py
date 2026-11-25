# ==========================================
# 解决 Mac 系统下 OpenMP 库冲突导致的崩溃
# 必须放在所有 import 的最前面！
# ==========================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

"""
CLIP 图文语义匹配模块 (M5)
负责人：黄涵
说明：
1. 内置超大中英失物词典，支持离线将任意常见中文翻译为英文。
2. 若词典未命中，自动降级使用在线翻译 API。
3. 采用 Batch Inference 批处理，速度极快。
"""

import torch
import clip
from PIL import Image

# ================= 超级中英词典（融合自前端） =================
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


# ================= 全局模型单例加载 =================
_GLOBAL_MODEL = None
_GLOBAL_PREPROCESS = None
_GLOBAL_DEVICE = None

def load_clip_model(model_name="ViT-B/32"):
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
        self.model, self.preprocess, self.device = load_clip_model(model_name)
        self.threshold = threshold

    def _format_text(self, text_query):
        """
        优化 Prompt：将中文翻译为英文（优先本地大词典，备选在线翻译），并包装成监控场景专用句子
        """
        text_query = str(text_query).strip()
        
        # 1. 检测是否包含中文字符
        if any('\u4e00' <= char <= '\u9fa5' for char in text_query):
            # A. 尝试本地超级大词典（长词优先，防止“双肩包”被“包”截断）
            matched_en = []
            temp_query = text_query
            for zh in sorted(ZH2EN.keys(), key=len, reverse=True):
                if zh in temp_query:
                    matched_en.append(ZH2EN[zh])
                    temp_query = temp_query.replace(zh, " ") # 替换掉匹配过的中文，防止重复
                    
            if matched_en:
                # 去重并拼接
                unique_en = list(set(matched_en))
                text_query = " ".join(unique_en)
            else:
                # B. 如果词典没有命中，降级尝试在线翻译
                try:
                    from deep_translator import GoogleTranslator
                    text_query = GoogleTranslator(source='auto', target='en').translate(text_query).lower()
                except Exception:
                    pass # 如果断网且没命中词典，只能原样返回（虽然后面识别率会很低，但程序不崩溃）
        
        # 2. 监控场景 Prompt 包装
        if "photo" not in text_query and "picture" not in text_query and "view" not in text_query:
            return f"a surveillance camera view of a {text_query} in a room"
        return text_query

    def match_images(self, text_query: str, candidate_images: list, top_k: int = 5):
        if text_query is None or not str(text_query).strip():
            return {"status": "error", "is_found": False, "message": "输入文本为空", "best_score": 0.0, "top_k_results": []}
        if not candidate_images:
            return {"status": "error", "is_found": False, "message": "未收到候选子图", "best_score": 0.0, "top_k_results": []}

        top_k = min(top_k, len(candidate_images))
        
        # 调用刚刚写的函数，完成翻译和包装！
        formatted_text = self._format_text(text_query)

        try:
            with torch.no_grad():
                # 文本特征提取
                text_input = clip.tokenize([formatted_text]).to(self.device)
                text_features = self.model.encode_text(text_input)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # 图像特征批量提取
                image_inputs = torch.stack([self.preprocess(img) for img in candidate_images]).to(self.device)
                image_features = self.model.encode_image(image_inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # 计算相似度
                similarity_scores = (100.0 * text_features @ image_features.T).squeeze(0)
                scores = (similarity_scores / 100.0).cpu().numpy().tolist()

            # 排序与聚合
            results = list(zip(candidate_images, scores))
            results.sort(key=lambda x: x[1], reverse=True)

            top_results = results[:top_k]
            best_score = top_results[0][1] if top_results else 0.0
            is_found = best_score >= self.threshold

            return {
                "status": "success",
                "is_found": is_found,
                "best_score": round(best_score, 4),
                "top_k_results": top_results,
                "message": "✅ 找到目标！" if is_found else "❌ 未找到目标（相似度未达到阈值）"
            }
            
        except Exception as e:
            return {"status": "error", "is_found": False, "message": f"图像编码或匹配失败: {str(e)}", "best_score": 0.0, "top_k_results": []}