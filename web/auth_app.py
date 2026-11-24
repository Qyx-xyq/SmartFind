import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from db_manager import db

def show_auth_page():
    # ==========================================
    # 1. 注入自定义 CSS，美化登录卡片
    # ==========================================
    st.markdown("""
        <style>
        /* 整体背景渐变 */
        .stApp {
            background: linear-gradient(135deg, #e0f2fe 0%, #f4faff 100%);
        }
        
        /* 隐藏自带的头部和脚部 */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* 登录卡片的主体样式 */
        .login-card {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(12px);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(31, 86, 153, 0.1);
            max-width: 420px;
            margin: 80px auto;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.5);
        }
        
        .login-title {
            font-size: 28px;
            font-weight: 800;
            color: #1d4ed8;
            margin-bottom: 5px;
            letter-spacing: 1px;
        }
        
        .login-subtitle {
            font-size: 14px;
            color: #64748b;
            margin-bottom: 30px;
        }
        
        /* 美化输入框 */
        .stTextInput > div > div > input {
            border-radius: 12px !important;
            border: 1.5px solid #cbd5e1 !important;
            padding: 12px 16px !important;
            font-size: 15px !important;
            background: #ffffff !important;
            transition: all 0.3s ease !important;
        }
        .stTextInput > div > div > input:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.15) !important;
        }
        
        /* 美化按钮 */
        .stButton > button, [data-testid="stFormSubmitButton"] > button {
            width: 100% !important;
            border-radius: 12px !important;
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 12px 0 !important;
            border: none !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease !important;
            margin-top: 10px;
        }
        .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 20px rgba(37, 99, 235, 0.3) !important;
        }
        
        /* 隐藏表单外部的边框 */
        [data-testid="stForm"] {
            border: none !important;
            padding: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # ==========================================
    # 2. 页面结构：居中卡片 + 选项卡
    # ==========================================
    col1, col2, col3 = st.columns([1, 2.5, 1])

    with col2:
        # 开启卡片容器
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<div class="login-title">🔍 慧眼寻物 SmartFind</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">基于 YOLO 与 CLIP 的智能失物招领系统</div>', unsafe_allow_html=True)

        # 选项卡
        tab1, tab2 = st.tabs(["🔐 账号登录", "✨ 新用户注册"])

        with tab1:
            with st.form("login_form"):
                account = st.text_input("账号", placeholder="请输入用户名 / 手机号 / 学号")
                password = st.text_input("密码", type="password", placeholder="请输入密码")
                submit_login = st.form_submit_button("立即登录")

                if submit_login:
                    success, info = db.login(account, password)
                    if success:
                        st.session_state['logged_in'] = True
                        st.session_state['user_info'] = info
                        st.success("登录成功！正在为您跳转...")
                        st.rerun()
                    else:
                        st.error(f"❌ {info}")

        with tab2:
            with st.form("register_form"):
                new_user = st.text_input("用户名", placeholder="设置您的登录昵称")
                new_phone = st.text_input("手机号", placeholder="请输入 11 位手机号")
                new_sid = st.text_input("学号", placeholder="请输入 8-12 位学号")
                new_pw = st.text_input("密码", type="password", placeholder="设置登录密码")
                
                # ================= 新增：内部邀请码 =================
                admin_code = st.text_input("管理员内部邀请码", type="password", placeholder="普通用户请留空")
                # ====================================================
                
                submit_reg = st.form_submit_button("注册账号")

                if submit_reg:
                    # 判断邀请码是否为 smartfind2026
                    role_to_assign = "admin" if admin_code == "smartfind2026" else "user"
                    
                    # 调用数据库注册（传入计算好的角色）
                    success, msg = db.register(new_user, new_phone, new_sid, new_pw, role=role_to_assign)
                    if success:
                        if role_to_assign == "admin":
                            st.success("🎉 管理员账号注册成功！请切换到登录选项卡。")
                        else:
                            st.success("✅ 注册成功！请切换到登录选项卡。")
                    else:
                        st.error(f"❌ {msg}")

        # 关闭卡片容器
        st.markdown('</div>', unsafe_allow_html=True)

# 如果单独测试这个文件，可以取消下面这行的注释
# if __name__ == "__main__":
#     st.set_page_config(page_title="登录", layout="wide")
#     show_auth_page()