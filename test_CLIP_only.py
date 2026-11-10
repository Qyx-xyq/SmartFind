"""
test_clip_only.py —— CLIP 模块独立调试脚本
作用：不依赖 YOLO，自己造几张图，专门测试 CLIP.py 的各种功能是否正常。
"""
from PIL import Image
from CLIP import CLIPMatcher

def run_tests():
    print("=== 开始单独测试 CLIP 模块 ===")
    
    # 1. 初始化你的 CLIP 模型（此时会打印“正在加载 CLIP 模型...”）
    # 阈值设成 0.20 方便测试
    matcher = CLIPMatcher(threshold=0.20)

    # 2. 造几个“假”的候选图（模拟 YOLO 切出来的图）
    # 这里为了直观，我直接生成纯色图。真实项目里，你可以用 Image.open("真实图片路径") 替换
    img_red = Image.new('RGB', (224, 224), color='red')
    img_blue = Image.new('RGB', (224, 224), color='blue')
    img_black = Image.new('RGB', (224, 224), color='black')
    
    # 这是一个包含 3 张候选图的列表，模拟 YOLO 的输出
    fake_yolo_crops = [img_red, img_blue, img_black]

    # ================= 测试场景 1：正常匹配 =================
    print("\n--- 场景 1：正常匹配（寻找红色的东西） ---")
    result1 = matcher.match_images("a red picture", fake_yolo_crops, top_k=3)
    
    print(f"是否找到: {result1['is_found']}")
    print(f"最高得分: {result1['best_score']}")
    print(f"提示信息: {result1['message']}")
    print("Top-K 排名:")
    for item in result1['top_k_results']:
        # item 包含 (图片对象, 得分)
        print(f"  - 得分: {item[1]:.4f}")

    # ================= 测试场景 2：查不到目标 =================
    print("\n--- 场景 2：查不到目标（找一只猫，但图里只有纯色） ---")
    result2 = matcher.match_images("a cat", fake_yolo_crops)
    print(f"是否找到: {result2['is_found']}")
    print(f"最高得分: {result2['best_score']}")
    print(f"提示信息: {result2['message']}")

    # ================= 测试场景 3：异常处理（传空文本） =================
    print("\n--- 场景 3：异常处理（用户没输入文字） ---")
    result3 = matcher.match_images("", fake_yolo_crops)
    print(f"状态: {result3['status']}")
    print(f"提示信息: {result3['message']}")

    # ================= 测试场景 4：异常处理（YOLO 啥也没切出来） =================
    print("\n--- 场景 4：异常处理（YOLO 返回了空列表） ---")
    result4 = matcher.match_images("a red picture", [])
    print(f"状态: {result4['status']}")
    print(f"提示信息: {result4['message']}")

if __name__ == "__main__":
    run_tests()