
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

def visualize_layout(data, img_dir):
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
        for layout in item['info']['layout']:
            if layout['segmentation']!=[]:
                segmentation = layout['segmentation']
                # segmentation=[[segmentation[2*i],segmentation[2*i+1]]for i in range(len(segmentation)//2)]
                cls_id=str(layout['cls'])
                layout_cls=data['meta_dicts']['layout_cls_map'][cls_id]
                rect = patches.Polygon(segmentation, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                plt.text(segmentation[0][0], segmentation[0][1] - 10, layout_cls, color='r', fontsize=12)
        plt.savefig('r.png', bbox_inches='tight')
        plt.close(fig)
        if i>10:
            break
with open('CDLA_DATASET/CDLA_train.json', 'r', encoding='utf-8') as f:
    unified_data = json.load(f)

# 定义图像文件所在目录
image_directory = "CDLA_DATASET"
visualize_layout(unified_data,image_directory)