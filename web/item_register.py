import sqlite3
import os
import numpy as np
from PIL import Image
# 假设你的 CLIP.py 在同一个目录下
from CLIP import load_clip_model 

DB_PATH = "smartfind.db"
UPLOAD_DIR = "uploads" # 用来存放用户上传的物品图片

# 确保上传目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)

def extract_image_feature(image_path):
    """使用 CLIP 提取图片特征向量"""
    import torch
    model, preprocess, device = load_clip_model()
    
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        # 归一化
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
    # 将 tensor 转换为 numpy 数组，再转为 bytes 存入 SQLite
    feature_numpy = image_features.cpu().numpy().astype(np.float32)
    return feature_numpy.tobytes()

def register_item(item_name, description, contact_info, image_file=None):
    """
    登记丢失物品
    :param image_file: 用户上传的图片文件（File对象），如果是空的就不存特征
    """
    image_path_to_db = None
    image_feature_bytes = None

    # 1. 如果有上传图片，保存图片并提取特征
    if image_file:
        # 保存图片到本地
        image_path_to_db = os.path.join(UPLOAD_DIR, image_file.name)
        with open(image_path_to_db, "wb") as f:
            f.write(image_file.getbuffer())
            
        print("正在提取图像特征（调用 CLIP）...")
        # 提取特征
        image_feature_bytes = extract_image_feature(image_path_to_db)
        print("✅ 特征提取完成！")

    # 2. 写入数据库
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO lost_item (item_name, description, contact_info, image_path, image_feature)
        VALUES (?, ?, ?, ?, ?)
    ''', (item_name, description, contact_info, image_path_to_db, image_feature_bytes))
    
    conn.commit()
    conn.close()
    print(f"✅ 物品 '{item_name}' 登记成功！")

# ================= 本地测试 =================
if __name__ == "__main__":
    # 测试1：不带图片的纯文本登记
    register_item("黑色双肩包", "遗失在图书馆三楼，侧面有一个白色水杯", "138xxxx")
    
    # 测试2：带图片的登记（假设你本地有一张 test.jpg）
    # with open("data/1.jpg", "rb") as f:  # 模拟前端上传的文件对象
    #     class MockFile:
    #         def __init__(self, file_obj, name):
    #             self.file_obj = file_obj
    #             self.name = name
    #         def getbuffer(self):
    #             return self.file_obj.read()
    #     mock_file = MockFile(f, "item_test.jpg")
    #     register_item("雨伞", "蓝色长柄伞", "QQ: 12345", image_file=mock_file)