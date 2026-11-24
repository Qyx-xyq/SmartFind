import streamlit as st
import os
import pandas as pd
import sys
from pathlib import Path

# 路径配置，确保能读到 db_manager
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_manager import db

# 页面标题
st.markdown("## ⚙️ 管理员控制台")
st.markdown("---")

# 定义图片存储目录
IMAGE_DIR = ROOT / "web" / "test_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# 使用 Tabs 分为图片管理和用户管理
tab1, tab2 = st.tabs(["🖼️ 监控图片管理", "👥 注册用户名单"])

# ================= Tab 1: 图片管理 =================
with tab1:
    st.markdown("### 📤 上传新照片")
    uploaded_files = st.file_uploader("选择图片", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if st.button("确认上传"):
        if uploaded_files:
            for f in uploaded_files:
                with open(IMAGE_DIR / f.name, "wb") as file:
                    file.write(f.getbuffer())
            st.success(f"成功上传 {len(uploaded_files)} 张照片！")
            st.rerun()
        else:
            st.warning("请先选择照片")

    st.markdown("---")
    st.markdown("### 🗑️ 删除现有照片")
    img_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not img_files:
        st.info("图库为空")
    else:
        cols = st.columns(4)
        for i, img_name in enumerate(img_files):
            with cols[i % 4]:
                st.image(str(IMAGE_DIR / img_name), caption=img_name, use_column_width=True)
                if st.button(f"删除 {img_name}", key=f"del_{img_name}"):
                    os.remove(IMAGE_DIR / img_name)
                    st.success("已删除")
                    st.rerun()

# ================= Tab 2: 用户名单 =================
with tab2:
    st.markdown("### 📋 所有注册用户")
    try:
        db.cursor.execute("SELECT user_id, username, phone, student_id, role FROM users")
        users = db.cursor.fetchall()
        if users:
            df = pd.DataFrame(users, columns=["ID", "用户名", "手机号", "学号", "角色"])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning("暂无注册用户")
    except Exception as e:
        st.error(f"读取数据库失败: {e}")