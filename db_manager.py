# db_manager.py
import sqlite3
import hashlib
import os
from datetime import datetime
import re

class DBManager:
    def __init__(self, db_path="data/smartfind.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _hash_password(self, password):
        """密码哈希加密"""
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    def _create_tables(self):
        """初始化建表"""
        # 1. 升级版用户表
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                phone TEXT UNIQUE NOT NULL,
                student_id TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user'
            )
        ''')
        
        # 2. 失物登记表 (保持不变)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS lost_items (
                item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                item_name TEXT NOT NULL,
                description TEXT,
                register_time TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            )
        ''')
        self.conn.commit()

    # ================== 用户业务 ==================
    def register(self, username, phone, student_id, password, role='user'):
        """注册：返回 (是否成功, 提示信息)"""
        # 基础校验
        if not all([username, phone, student_id, password]):
            return False, "所有带*号的必填项不能为空"
        if not re.match(r"^1[3-9]\d{9}$", phone):
            return False, "手机号格式不正确"
        if not re.match(r"^\d{8,12}$", student_id):
            return False, "学号格式不正确（应为8-12位数字）"
        
        try:
            hashed_pw = self._hash_password(password)
            self.cursor.execute(
                "INSERT INTO users (username, phone, student_id, password_hash, role) VALUES (?, ?, ?, ?, ?)",
                (username, phone, student_id, hashed_pw, role)
            )
            self.conn.commit()
            return True, "注册成功"
        except sqlite3.IntegrityError as e:
            # 根据报错信息判断哪个字段重复了
            error_msg = str(e)
            if "users.username" in error_msg: return False, "用户名已存在"
            if "users.phone" in error_msg: return False, "手机号已被注册"
            if "users.student_id" in error_msg: return False, "学号已被注册"
            return False, "注册失败：信息冲突"
        except Exception as e:
            return False, f"注册异常: {str(e)}"

    def login(self, account, password):
        """
        多方式登录：account 可以是用户名、手机号或学号
        返回 (是否成功, 用户信息字典 或 错误提示)
        """
        if not account or not password:
            return False, "账号和密码不能为空"
        
        hashed_pw = self._hash_password(password)
        
        # 核心：OR 查询，匹配任意一个字段即可
        self.cursor.execute('''
            SELECT user_id, username, role FROM users 
            WHERE (username = ? OR phone = ? OR student_id = ?) AND password_hash = ?
        ''', (account, account, account, hashed_pw))
        
        user = self.cursor.fetchone()
        if user:
            return True, {
                "user_id": user[0], 
                "username": user[1], 
                "role": user[2]  # 返回角色，用于校验是否为管理员
            }
        return False, "账号或密码错误"

    # ================== 物品业务 ==================
    def add_lost_item(self, user_id, item_name, description):
        """登记物品"""
        if not item_name.strip():
            return False, "物品名称不能为空"
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cursor.execute(
                "INSERT INTO lost_items (user_id, item_name, description, register_time) VALUES (?, ?, ?, ?)",
                (user_id, item_name, description, now)
            )
            self.conn.commit()
            return True, "物品登记成功"
        except Exception as e:
            return False, f"登记异常: {str(e)}"

    def get_user_items(self, user_id):
        """查询该用户登记过的所有物品"""
        self.cursor.execute(
            "SELECT item_name, description, register_time FROM lost_items WHERE user_id = ? ORDER BY register_time DESC",
            (user_id,)
        )
        return self.cursor.fetchall()

    def close(self):
        self.conn.close()

db = DBManager()