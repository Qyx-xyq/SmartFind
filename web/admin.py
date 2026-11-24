import streamlit as st
import os
import shutil
from PIL import Image
import pandas as pd
import sys
from pathlib import Path

# ================= 路径与导入设置 =================
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 导入同学的数据库模块
try:
    from db_manager import db
except ImportError:
    st.error("找不到 db_manager.py，请确保它在项目根目录下！")
    st.stop()

# ================= 页面配置与权限校验 =================
# 注意：因为这是子页面，如果 main.py 已经调了 set_page_config，这行要注释掉
# st.set_page_config(page_title="管理员后台 - 慧眼寻物", page_icon="⚙️", layout="wide")

# 拦截越权访问
if not st.session_state.get('logged_in') or st.session_state.get('user_info', {}).get('role') != 'admin':
    st.error("⛔ 权限不足：只有管理员才能访问此页面。")
    st.stop()

admin_name = st.session_state['user_info']['username']

# ================= 自定义美化样式 =================
st.markdown("""
<style>
    .admin-header { font-size: 28px; font-weight: bold; color: #2f74d0; margin-bottom: 20px;}
    .card { background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 15px rgba(31,86,153,.1); margin-bottom: 20px;}
    /* 隐藏原生的菜单，因为外层有 main.py 了 */
    #MainMenu {visibility: hidden;} 
</style>
""", unsafe_allow_html=True)

st.markdown(f'<div class="admin-header">👑 管理员控制台 (当前登录: {admin_name})</div>', unsafe_allow_html=True)

# 定义监控图片的存放目录
IMAGE_DIR = os.path.join("web", "test_images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# 使用 Tabs 标签页把功能分开
tab_images, tab_users = st.tabs(["🖼️ 监控图片管理", "👥 注册用户管理"])

# ================= Tab 1: 图片管理 =================
with tab_images:
    st.markdown("### 📤 上传新的监控照片")
    
    uploaded_files = st.file_uploader("选择要上传的图片 (支持 JPG, PNG)", 
                                      type=["jpg", "jpeg", "png"], 
                                      accept_multiple_files=True)
    
    if st.button("确认上传", type="primary"):
        if uploaded_files:
            success_count = 0
            for uploaded_file in uploaded_files:
                file_path = os.path.join(IMAGE_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                success_count += 1
            st.success(f"✅ 成功上传 {success_count} 张图片到图库！")
            st.rerun()
        else:
            st.warning("请先选择要上传的图片。")

    st.divider()
    
    st.markdown("### 🗑️ 现有图库管理 (删除照片)")
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        st.info("图库当前为空。")
    else:
        st.write(f"图库中共有 **{len(image_files)}** 张图片。")
        cols = st.columns(4)
        for idx, img_name in enumerate(image_files):
            with cols[idx % 4]:
                img_path = os.path.join(IMAGE_DIR, img_name)
                try:
                    img = Image.open(img_path)
                    st.image(img, caption=img_name, use_container_width=True)
                    if st.button(f"删除 {img_name}", key=f"del_{img_name}"):
                        os.remove(img_path)
                        st.success(f"已删除 {img_name}")
                        st.rerun()
                except Exception as e:
                    st.error(f"读取图片 {img_name} 失败: {e}")

# ================= Tab 2: 用户管理 (接数据库) =================
with tab_users:
    st.markdown("### 📋 系统注册用户名单")
    st.info("💡 以下数据实时读取自 SQLite 数据库。")
    
    try:
        # 查数据库
        db.cursor.execute("SELECT user_id, username, phone, student_id, role FROM users ORDER BY user_id ASC")
        all_users = db.cursor.fetchall()
        
        if not all_users:
            st.warning("当前系统还没有注册用户。")
        else:
            # 整理成表格数据
            user_list = []
            for u in all_users:
                user_list.append({
                    "用户ID": u[0],
                    "用户名": u[1],
                    "手机号": u[2],
                    "学号": u[3],
                    "角色": "👑 管理员" if u[4] == "admin" else "普通用户"
                })
            
            # 展示表格
            df = pd.DataFrame(user_list)
            st.dataframe(
                df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "用户ID": st.column_config.NumberColumn(width="small"),
                    "角色": st.column_config.TextColumn(width="medium")
                }
            )
            
            st.caption(f"统计：共 {len(user_list)} 名用户，其中管理员 {len(df[df['角色'] == '👑 管理员'])} 名。")
            
    except Exception as e:
        st.error(f"读取数据库失败：{e}")