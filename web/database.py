import sqlite3
import os

DB_PATH = "smartfind.db"

def init_db():
    """初始化数据库，创建丢失物品表"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 创建 lost_item 表 (根据你们的需求文档设计)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lost_item (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_name TEXT NOT NULL,
            description TEXT,
            image_path TEXT,       -- 本地保存的图片路径
            image_feature BLOB,    -- 存放 CLIP 提取的图像特征(转为二进制)，进阶功能
            contact_info TEXT,     -- 失主联系方式
            status TEXT DEFAULT '未找回', -- 状态：未找回/已找回
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ 数据库和物品表初始化完成！")

if __name__ == "__main__":
    init_db()