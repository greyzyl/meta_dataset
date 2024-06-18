import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

def visualize_dense_description(data):
    for item in data['data']:
        dense_descri=item['info']['dense_description']
        xml_type=dense_descri['xml_type']
        xml=dense_descri['xml']
        output_path = 'output.'+xml_type
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml)
def visualize_ocr(data, img_dir):
    """
    可视化 OCR 和 KIE 部分的数据。
    """
    for item in data['data']:
        img_path = Path(os.path.join(img_dir, item['img_meta']['img_path']))
        if not img_path.exists():
            print(f"Image {img_path} not found.")
            continue

        img = Image.open(img_path)
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # 可视化 OCR 部分
        for ocr in item['info']['ocr']['word_level']:
            if ocr['bbox']!=[]:
                bbox = ocr['bbox']
                rect = patches.Polygon([[bbox[0],bbox[1]],[bbox[2],bbox[3]],[bbox[4],bbox[5]],[bbox[6],bbox[7]]], linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                plt.text(bbox[0], bbox[1] - 10, ocr['text'], color='r', fontsize=12)
        plt.savefig('r.png', bbox_inches='tight')
        plt.close(fig)

# 读取转换后的数据
with open('PubTabNet/PubTabNet_example.json', 'r', encoding='utf-8') as f:
    unified_data = json.load(f)

# 定义图像文件所在目录
image_directory = "/home/yuhaiyang/zyl/mllm/data/PubTabNet/image/train"

# 可视化数据
visualize_ocr(unified_data, image_directory)
visualize_dense_description(unified_data)
