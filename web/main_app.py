import streamlit as st
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auth_app import show_auth_page
import smartfind_app

st.set_page_config(page_title="慧眼寻物 SmartFind", page_icon="🔍", layout="centered")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    show_auth_page()
else:
    user = st.session_state.get('user_info') or {}
    username = user.get('username', '未知用户')
    role = user.get('role', 'user')
    
    # 侧边栏
    st.sidebar.success(f"👤 欢迎, {username}")
    st.sidebar.info(f"身份：{'👑 管理员' if role == 'admin' else '👤 普通用户'}")
    if st.sidebar.button("🚪 退出登录", use_container_width=True):
        st.session_state['logged_in'] = False
        st.session_state['user_info'] = None
        st.rerun()

    # ================= 核心修改：用 Tabs 切换页面 =================
    if role == 'admin':
        # 管理员显示两个标签页
        tab_search, tab_admin = st.tabs(["🔍 智能检索", "⚙️ 管理员后台"])
        
        with tab_search:
            smartfind_app.main()
            
        with tab_admin:
            # 动态读取并执行 admin.py
            try:
                admin_file = _HERE / "admin.py"
                with open(admin_file, "r", encoding="utf-8") as f:
                    code = f.read()
                # 防止冲突
                code = code.replace("st.set_page_config", "# st.set_page_config")
                exec(code, globals())
            except Exception as e:
                st.error(f"加载管理后台失败: {e}")
    else:
        # 普通用户只有检索
        smartfind_app.main()