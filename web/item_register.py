import streamlit as st
import sys
from pathlib import Path
import os

# 确保能导入根目录的 db_manager
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from db_manager import db

def show_register_page():
    """显示物品登记页面"""
    st.title("📝 失物登记中心")
    st.markdown("请填写丢失物品的信息，方便系统或管理员为您匹配。")
    st.divider()

    # 如果用户没有登录，提示去登录
    if 'user_info' not in st.session_state or not st.session_state['user_info']:
        st.warning("⚠️ 请先登录后再进行物品登记！")
        return

    current_user = st.session_state['user_info']
    
    # 使用表单，避免用户每次输入都刷新页面
    with st.form("register_form"):
        item_name = st.text_input("物品名称 *", placeholder="例如：黑色双肩包、透明雨伞")
        
        description = st.text_area("详细描述 *", placeholder="例如：遗失在食堂二楼，侧面有一个白色挂件...")
        
        contact_info = st.text_input("联系方式", placeholder="手机号、微信号或QQ（选填）")
        
        image_file = st.file_uploader("上传物品照片（选填，支持 jpg/png）", type=["jpg", "jpeg", "png"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("确认登记", use_container_width=True)

    # 处理提交逻辑
    if submit_btn:
        if not item_name.strip() or not description.strip():
            st.error("❌ 物品名称和详细描述为必填项！")
        else:
            with st.spinner("正在保存登记信息..."):
                # 调用你的 db_manager.py 中的 add_lost_item 方法
                # 注意：你的 db_manager 里目前没有 contact_info 和 image 的处理，下面教你怎么改
                success, msg = db.add_lost_item(
                    user_id=current_user['user_id'],
                    item_name=item_name,
                    description=description,
                    contact_info=contact_info,
                    image_file=image_file
                )
                
                if success:
                    st.success(f"✅ {msg}")
                    st.balloons() # 放个气球庆祝一下
                else:
                    st.error(f"❌ 登记失败：{msg}")

    st.divider()
    
    # 展示当前用户的历史登记记录
    st.subheader("📋 我的登记记录")
    my_items = db.get_user_items(current_user['user_id'])
    
    if not my_items:
        st.info("您还没有登记过任何物品。")
    else:
        # 用表格展示
        import pandas as pd
        df = pd.DataFrame(my_items, columns=["记录ID","物品名称", "描述", "联系方式", "图片路径", "状态", "登记时间"])
        st.dataframe(df, use_container_width=True, hide_index=True)

# 如果单独测试这个文件
if __name__ == "__main__":
    st.set_page_config(page_title="物品登记", layout="centered")
    show_register_page()