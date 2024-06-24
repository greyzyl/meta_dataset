import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

def visualize_ocr(data, img_dir):
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
        for ner in item['info']["kie"]["NER"]:
            if ner["ocr_info"]['all']['bbox']!=[]:
                bbox = ner["ocr_info"]['all']['bbox']
                ner_cls=data["meta_dicts"]["NER_cls_map"][str(ner['category'])]
                rect = patches.Rectangle(bbox[0],bbox[1][0]-bbox[0][0],bbox[1][1]-bbox[0][1], linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                plt.text(bbox[0][0], bbox[0][1], ner_cls['describe'], color='b', fontsize=5)
        plt.axis('off')
        plt.savefig('r.png', bbox_inches='tight', dpi=300)
        plt.close(fig)
        if i>0:
            break

# 读取转换后的数据
with open('CORD/train/cord_train_example.json', 'r', encoding='utf-8') as f:
    unified_data = json.load(f)

# 定义图像文件所在目录
image_directory = "CORD"

# 可视化数据
visualize_ocr(unified_data, image_directory)