import streamlit as st
from PIL import Image, ImageDraw
import os

# ================= 1. 页面基本设置 =================
st.set_page_config(page_title="慧眼寻物 SmartFind", page_icon="🔍", layout="wide")

st.title("🔍 慧眼寻物 SmartFind 智能失物检索系统")
st.write("欢迎使用！请输入您丢失的物品描述，系统将从监控画面中为您寻找。")

# ================= 2. 侧边栏设置 =================
with st.sidebar:
    st.header("⚙️ 系统设置")
    top_k = st.slider("返回最匹配的图片数量", min_value=1, max_value=5, value=3)
    st.info("当前模式：演示模式（使用假数据）")

# ================= 3. 顶部输入区 =================
# 用 columns 把输入框和按钮放在同一行
col_input, col_btn = st.columns([4, 1])

with col_input:
    # 文本输入框
    query = st.text_input("请输入丢失物品描述", placeholder="例如：黑色背包、笔记本电脑...")

with col_btn:
    st.write("") # 占位，为了对齐
    st.write("") # 占位，为了对齐
    # 搜索按钮
    search_clicked = st.button("开始查找 🔍", use_container_width=True, type="primary")

# ================= 4. 处理搜索逻辑与展示结果 =================
if search_clicked:
    if not query.strip():
        st.warning("⚠️ 请先输入物品描述再点击查找！")
    else:
        st.success(f"✅ 正在为您检索：**{query}**，请稍候...")
        st.markdown("---")
        
        # ---------------------------------------------------------
        # ⚠️ 这里就是以后要换成 YOLO 和 CLIP 同学代码的地方！
        # 现在我们先写死一个“假结果”让网页跑起来
        # ---------------------------------------------------------
        fake_results = [
            {
                "image_path": "web/test_images/001.jpg", # 替换成你文件夹里真实的图片名
                "box": [100, 80, 400, 350],              # 模拟 YOLO 框出的坐标 [x1, y1, x2, y2]
                "clip_score": 0.92                       # 模拟 CLIP 计算的相似度得分
            },
            {
                "image_path": "web/test_images/001.jpg", 
                "box": [500, 150, 700, 400],             
                "clip_score": 0.78                       
            }
        ]
        # ---------------------------------------------------------

        # 展示结果
        st.subheader(f"🎯 找到 {len(fake_results)} 个相关结果")
        
        # 使用 columns 分栏展示，每行排 2 张图
        cols = st.columns(2)
        
        for idx, res in enumerate(fake_results):
            # 计算当前图片该放在哪一列
            col = cols[idx % 2] 
            
            with col:
                with st.container(border=True): # 加个边框更好看
                    try:
                        # 1. 读取原图
                        img = Image.open(res["image_path"])
                        
                        # 2. 用 PIL 画红框
                        draw = ImageDraw.Draw(img)
                        x1, y1, x2, y2 = res["box"]
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
                        
                        # 3. 显示图片
                        st.image(img, caption=f"结果 {idx+1}：{res['image_path']}", use_container_width=True)
                        
                        # 4. 显示分数
                        st.metric(label="CLIP 图文相似度", value=f"{res['clip_score']:.2%}")
                        
                    except FileNotFoundError:
                        st.error(f"图片找不到：{res['image_path']}。请检查路径是否正确。")
