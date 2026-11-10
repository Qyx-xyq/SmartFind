# SmartFind —— 智慧寻物系统
功能简介：公共场所遗失物品频发，人工翻看监控查找失物效率低、耗人力。我们想借助图像识别，自动匹配监控画面与遗失物品，减轻工作人员负担，更快帮失主找回物品。

#### 文件介绍：
##### ├──demo_pipeline 跑通YOLO-CLIP流水线
##### | |- CLIP.py       CLIP模型示意
##### | |- YOLI.py       YOLO模型示意
##### | |- pipeline.py   简易YOLO-CLIP流水线示意
##### | |- data          输入/输出的示例图片
##### |
##### ├──CLIP.py       CLIP模型
##### ├──YOLO.py       YOLO模型
##### ├──train_yolo.py   YOLO 训练脚本（需要画框的 txt 标签）
##### ├── train_clip.py  CLIP 训练脚本（只需要 图片+文字描述）
##### ├── main.py        最终流水线入口
##### ├──data/           存放两套不同的数据集
##### └──moedels         模型存储
##### |└──yolov8n.pt     yolov8官方预训练模型
