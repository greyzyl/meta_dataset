import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

def visualize_data(data, img_dir):
    """
    可视化 OCR 和 KIE 部分的数据。
    """
    for item in data['data']:
        img_path = Path(img_dir) / item['img_path']
        if not img_path.exists():
            print(f"Image {img_path} not found.")
            continue

        img = Image.open(img_path)
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # 可视化 OCR 部分
        for ocr in item['info']['ocr']['word_level']:
            bbox = ocr['bbox']
            rect = patches.Polygon([[bbox[0],bbox[1]],[bbox[2],bbox[3]],[bbox[4],bbox[5]],[bbox[6],bbox[7]]], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(bbox[0], bbox[1] - 10, ocr['text'], color='r', fontsize=12)

        # 可视化 KIE 部分的 NER
        for ner in item['info']['kie']['NER']:
            bbox = ner['ocr_info']['all']['bbox']
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
            plt.text(bbox[0], bbox[1] - 10, ner['ocr_info']['all']['text'], color='b', fontsize=12)

        plt.savefig('r.png', bbox_inches='tight')
        plt.close(fig)

# 读取转换后的数据
with open('CORD/train/cord_train.json', 'r', encoding='utf-8') as f:
    unified_data = json.load(f)

# 定义图像文件所在目录
image_directory = "CORD"

# 可视化数据
visualize_data(unified_data, image_directory)
