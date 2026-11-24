import streamlit as st
import sys
from pathlib import Path

# ============================================================================
# 1. 路径与环境配置
# ============================================================================
# 获取当前文件所在的目录 (即 web/ 目录) 和项目根目录
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent

# 确保项目根目录在系统路径中，以便能 import 到外层的 YOLO.py 等文件
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 导入你写的登录页，和同学写的检索页
from auth_app import show_auth_page
import smartfind_app # 这里导入同学的代码

# ============================================================================
# 2. 页面基础配置（必须在所有 st 命令之前调用，且全局只能调用一次）
# ============================================================================
st.set_page_config(page_title="慧眼寻物 SmartFind", page_icon="🔍", layout="centered")

# 1. 检查登录状态
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    # 没登录，只显示登录页面
    show_auth_page()

# ----------------- 已登录：显示业务主界面 -----------------
else:
    # 登录了，显示顶部欢迎语和退出按钮
    user = st.session_state['user_info']
    st.sidebar.success(f"欢迎, {user['username']} ({user['role']})")
    if st.sidebar.button("退出登录"):
        st.session_state['logged_in'] = False
        st.session_state['user_info'] = None
        st.rerun()
        
    # 显示同学的检索界面 (调他的 main 函数)
    # 注意：同学的代码里如果用了 st.set_page_config，可能会报错，需要让他把那行删掉，或者改成 try-except
    smartfind_app.main() 