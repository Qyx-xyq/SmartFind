import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from auth_app import show_auth_page
from item_register import show_register_page  # 导入物品登记页
import smartfind_app                         # 导入物品检索页

st.set_page_config(page_title="慧眼寻物", layout="centered")

if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    show_auth_page()
else:
    user = st.session_state['user_info']
    
    with st.sidebar:
        st.markdown(f"### 👤 欢迎, {user['username']}")
        st.markdown(f"身份: `{user['role']}`")
        st.divider()
        
        # 导航菜单
        page = st.radio("导航菜单", ["🔍 搜索失物", "📝 登记物品"])
        
        st.divider()
        if st.button("退出登录", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['user_info'] = None
            st.rerun()

    # 根据用户选择，渲染不同的页面
    if page == "🔍 搜索失物":
        smartfind_app.main()  # 渲染同学的页面
    elif page == "📝 登记物品":
        show_register_page()  # 渲染你的物品登记页