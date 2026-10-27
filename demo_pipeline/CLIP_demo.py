import torch
import clip
from PIL import Image

# 1. 自动选择设备（GPU 或 CPU）
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前使用的设备: {device}")

# 2. 加载官方模型和预处理函数（首次运行会自动下载）
# 这里选择最轻量的 ViT-B/32 模型
model, preprocess = clip.load("ViT-B/32", device=device)

# 3. 准备测试图片
# 为了让你马上就能跑通，这里生成一张纯蓝色的测试图。
# 等验证通过后，你可以换成真实图片：image = Image.open("你的图片.jpg")
image = Image.new('RGB', (224, 224), color='blue')

# 4. 准备文本描述
# 注意：官方库是用 clip.tokenize 来处理文字的
text_descriptions = ["a photo of a bag", "a photo of a cat", "a photo of a car"]
text = clip.tokenize(text_descriptions).to(device)

# 5. 特征提取与相似度计算
with torch.no_grad():
    # 预处理图片（缩放、归一化等），并增加 batch 维度
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    # 提取图像和文本特征
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text)
    
    # 计算相似度：将特征归一化后进行点积
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 乘以 100 是为了放大差异，softmax 把得分转化为概率
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# 6. 打印结果
print("\n--- 验证结果 ---")
print("候选文本描述:", text_descriptions)
print("匹配概率:", similarity.cpu().numpy()[0])