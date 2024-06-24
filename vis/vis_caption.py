import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path


def visualize_caption(data, img_dir):
    """
    可视化 OCR 和 KIE 部分的数据。
    """
    for i,item in enumerate(data['data']):
        img_path = Path(os.path.join(img_dir, item['img_meta']['img_path']))
        if not img_path.exists():
            print(f"Image {img_path} not found.")
            continue

        img = Image.open(img_path).convert('RGB')
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # 可视化 OCR 部分
        for i,caption in enumerate(item['info']["caption"]):
            if caption['confidence']=='yes':
                plt.text(0,i*60 , caption['caption'], color='r', fontsize=5)
            elif caption['confidence']=='normal':
                plt.text(0,i*60 , caption['caption'], color='b', fontsize=5)
        plt.savefig('r.png', dpi=300,bbox_inches='tight')
        plt.close(fig)
        if i>0:
            break

# 读取转换后的数据
with open('TextCaps/TextCaps_train_example.json', 'r', encoding='utf-8') as f:
    unified_data = json.load(f)

# 定义图像文件所在目录
image_directory = "TextCaps"

# 可视化数据
visualize_caption(unified_data, image_directory)