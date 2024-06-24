import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
os.makedirs(os.path.join('work_dir','vis'),exist_ok=True)
def visualize_ocr_word(data, img_dir):
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
        for ocr in item['info']['ocr']['word_level']:
            if ocr['bbox']!=[]:
                bbox = ocr['bbox']
                rect = patches.Polygon(bbox, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                plt.text(bbox[0][0], bbox[0][1] , ocr['text'], color='b', fontsize=12)
        plt.savefig('r.png', dpi=300,bbox_inches='tight')
        plt.close(fig)
        if i>0:
            break

def visualize_ocr_line(data, img_dir):
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
        for ocr in item['info']['ocr']['line_level']:
            if ocr['bbox']!=[]:
                bbox = ocr['bbox']
                rect = patches.Polygon(bbox, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                plt.text(bbox[0][0], bbox[0][1] , ocr['text'], color='b', fontsize=12)
        plt.savefig(os.path.join('work_dir','vis',f'r{i}.png'), dpi=300,bbox_inches='tight')
        plt.close(fig)
        # if i>0:
        #     break

# 读取转换后的数据
with open('TextCaps/TextCaps_train_example.json', 'r', encoding='utf-8') as f:
    unified_data = json.load(f)

# 定义图像文件所在目录
image_directory = "TextCaps"

# 可视化数据
visualize_ocr_line(unified_data, image_directory)