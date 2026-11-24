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

# 导入各个模块
try:
    from auth_app import show_auth_page
    import smartfind_app
except ImportError as e:
    st.error(f"模块导入失败: {e}\n请检查 auth_app.py 和 smartfind_app.py 是否在 web 文件夹下。")
    st.stop()

# ============================================================================
# 2. 页面基础配置（必须在所有 st 命令之前调用，且全局只能调用一次）
# ============================================================================
st.set_page_config(page_title="慧眼寻物 SmartFind", page_icon="🔍", layout="centered")

# ============================================================================
# 3. 登录状态判断与页面路由
# ============================================================================
# 初始化 session_state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_info' not in st.session_state:
    st.session_state['user_info'] = None

# ----------------- 未登录：显示登录/注册页 -----------------
if not st.session_state['logged_in']:
    show_auth_page()

# ----------------- 已登录：显示业务主界面 -----------------
else:
    # 获取用户信息 (字典格式，包含 username, role 等)
    user = st.session_state.get('user_info') or {}
    username = user.get('username', '未知用户')
    role = user.get('role', 'user')
    
    # ================= 侧边栏公共区域 =================
    st.sidebar.success(f"👤 欢迎, {username}")
    
    # 如果是管理员，在侧边栏显示身份标识
    if role == 'admin':
        st.sidebar.info("身份：👑 管理员")
    else:
        st.sidebar.info("身份：👤 普通用户")
        
    # 侧边栏退出按钮
    if st.sidebar.button("🚪 退出登录", use_container_width=True):
        st.session_state['logged_in'] = False
        st.session_state['user_info'] = None
        st.rerun()  # 刷新页面，回到登录状态

    # ================= 侧边栏导航菜单 =================
    if role == 'admin':
        # 管理员显示两个选项
        page = st.sidebar.radio("导航菜单", ["🔍 智能检索", "⚙️ 管理后台"])
    else:
        # 普通用户只能进入检索页
        page = "🔍 智能检索"

    # ================= 页面内容分发 =================
    if page == "🔍 智能检索":
        # 调用 smartfind_app.py 里的 main 函数
        smartfind_app.main()
        
    elif page == "⚙️ 管理后台":
        # 动态执行 web/admin.py（避免 import 时的 set_page_config 冲突）
        try:
            admin_file = _HERE / "admin.py"
            if not admin_file.exists():
                st.error("❌ 找不到 web/admin.py 文件，请检查文件是否存在。")
            else:
                with open(admin_file, "r", encoding="utf-8") as f:
                    code = f.read()
                # 拦截可能存在的 set_page_config 冲突
                code = code.replace("st.set_page_config", "# st.set_page_config")
                # 执行 admin.py 的代码
                exec(code, globals())
        except Exception as e:
            st.error(f"❌ 加载管理后台失败: {e}")