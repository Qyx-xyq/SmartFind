import streamlit as st
import sys
from pathlib import Path

# 确保能导入根目录的 db_manager
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from db_manager import db

def show_auth_page():
    st.title("🔐 慧眼寻物 - 身份认证")
    
    # 用选项卡区分登录和注册
    tab1, tab2 = st.tabs(["登录", "注册"])
    
    with tab1:
        st.subheader("账号登录")
        account = st.text_input("账号（用户名/手机号/学号）", key="login_account")
        password = st.text_input("密码", type="password", key="login_pw")
        
        if st.button("登录", use_container_width=True):
            success, info = db.login(account, password)
            if success:
                st.session_state['logged_in'] = True
                st.session_state['user_info'] = info
                st.success("登录成功！正在跳转...")
                st.rerun() # 刷新页面，触发跳转
            else:
                st.error(info)
                
    with tab2:
        st.subheader("新用户注册")
        new_user = st.text_input("用户名", key="reg_user")
        new_phone = st.text_input("手机号", key="reg_phone")
        new_sid = st.text_input("学号", key="reg_sid")
        new_pw = st.text_input("密码", type="password", key="reg_pw")
        
        if st.button("提交注册", use_container_width=True):
            success, msg = db.register(new_user, new_phone, new_sid, new_pw)
            if success:
                st.success("注册成功！请切换到登录页登录。")
            else:
                st.error(msg)