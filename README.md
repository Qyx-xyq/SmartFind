# SmartFind —— 智慧寻物系统
功能简介：公共场所遗失物品频发，人工翻看监控查找失物效率低、耗人力。我们想借助图像识别，自动匹配监控画面与遗失物品，减轻工作人员负担，更快帮失主找回物品。

#### 文件介绍：
##### |- demo_pipeline 跑通YOLO-CLIP流水线
##### | |- CLIP.py       CLIP模型示意
##### | |- YOLI.py       YOLO模型示意
##### | |- pipeline.py   简易YOLO-CLIP流水线示意
##### | |- data          输入/输出的示例图片
##### |
##### |- CLIP.py       CLIP模型
##### |- YOLI.py       YOLO模型
##### |- train.py      模型训练入口（待定）
##### |- main.py       主功能入口（待定）
##### |- data          输入/输出数据
##### |
##### |- weights       模型权重存储
##### | |- yolov8n.pt    yolov8官方预训练模型权重