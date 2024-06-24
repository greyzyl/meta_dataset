import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path


def visualize_vqa(data, img_dir):
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
        for i,vqa in enumerate(item['info']["vqa"]):
            plt.text(0,i*60 , vqa['question'], color='r', fontsize=5)
            for j,answer in enumerate(vqa['answers']):
                plt.text(0,i*60+(j+1)*30 , answer['answer'], color='b', fontsize=5)
        plt.savefig('r.png', dpi=300,bbox_inches='tight')
        plt.close(fig)
        if i>0:
            break

# 读取转换后的数据
with open('DocVQA/DocVQA_train_example.json', 'r', encoding='utf-8') as f:
    unified_data = json.load(f)

# 定义图像文件所在目录
image_directory = "DocVQA"

# 可视化数据
visualize_vqa(unified_data, image_directory)