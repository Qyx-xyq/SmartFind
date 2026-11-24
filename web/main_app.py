import streamlit as st
import sys
from pathlib import Path

# 解决模块导入路径问题
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 假设你的认证页面在 auth_app.py，检索在 smartfind_app.py，后台在 admin.py
from auth_app import show_auth_page
import smartfind_app 

# ⚠️ 注意：这里不能写 st.set_page_config，必须放到你的 auth_app 里，或者放在最顶端，
# 但既然你们分文件了，建议把这个配置写进 auth_app.py 的顶部。

# 1. 检查登录状态
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    # 没登录，显示登录/注册页面
    show_auth_page()
else:
    # 已经登录
    user = st.session_state['user_info']
    
    # 侧边栏：欢迎信息和退出
    st.sidebar.success(f"欢迎, {user['username']}")
    st.sidebar.info(f"身份：{'👑 管理员' if user['role'] == 'admin' else '👤 普通用户'}")
    
    # 侧边栏导航菜单
    if user['role'] == 'admin':
        # 如果是管理员，给他两个选项
        page = st.sidebar.radio("导航菜单", ["🔍 智能检索", "⚙️ 管理后台"])
    else:
        # 普通用户只能选检索
        page = "🔍 智能检索"

    if st.sidebar.button("退出登录", use_container_width=True):
        st.session_state['logged_in'] = False
        st.session_state['user_info'] = None
        st.rerun()

    # 2. 根据选择的路由显示不同的页面
    if page == "🔍 智能检索":
        # 调用同学的检索代码
        try:
            smartfind_app.main()
        except AttributeError:
            st.error("启动失败：请确保 smartfind_app.py 中有一个 main() 函数，并且没有在最外层直接执行 UI 代码。")
            
    elif page == "⚙️ 管理后台":
        # 调用你刚写的管理员代码
        # 这里我们用 exec 动态执行 admin.py（最简单的多页面合并方式）
        try:
            with open("web/admin.py", "r", encoding="utf-8") as f:
                code = f.read()
            # 拦截 admin.py 中的 set_page_config 防止报错
            code = code.replace("st.set_page_config", "# st.set_page_config")
            exec(code)
        except FileNotFoundError:
            st.error("找不到 web/admin.py 文件。")