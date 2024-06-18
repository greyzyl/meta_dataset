import json
import os
from pathlib import Path

import cv2
NER_cls_map_id = {
    0: {
        "category": "menu.nm",
        "describe": "name of menu"
    },
    1: {
        "category": "menu.num",
        "describe": "identification # of menu"
    },
    2: {
        "category": "menu.unitprice",
        "describe": "unit price of menu"
    },
    3: {
        "category": "menu.cnt",
        "describe": "quantity of menu"
    },
    4: {
        "category": "menu.discountprice",
        "describe": "discounted price of menu"
    },
    5: {
        "category": "menu.price",
        "describe": "total price of menu"
    },
    6: {
        "category": "menu.itemsubtotal",
        "describe": "price of each menu after discount applied"
    },
    7: {
        "category": "menu.vatyn",
        "describe": "whether the price includes tax or not"
    },
    8: {
        "category": "menu.etc",
        "describe": "others"
    },
    9: {
        "category": "menu.sub_nm",
        "describe": "name of submenu"
    },
    10: {
        "category": "menu.sub_unitprice",
        "describe": "unit price of submenu"
    },
    11: {
        "category": "menu.sub_cnt",
        "describe": "quantity of submenu"
    },
    12: {
        "category": "menu.sub_price",
        "describe": "total price of submenu"
    },
    13: {
        "category": "menu.sub_etc",
        "describe": "others"
    },
    14: {
        "category": "void_menu.nm",
        "describe": "name of menu"
    },
    15: {
        "category": "void_menu.price",
        "describe": "total price of menu"
    },
    16: {
        "category": "sub_total.subtotal_price",
        "describe": "sub_total price"
    },
    17: {
        "category": "sub_total.discount_price",
        "describe": "discounted price in total"
    },
    18: {
        "category": "sub_total.service_price",
        "describe": "service charge"
    },
    19: {
        "category": "sub_total.othersvc_price",
        "describe": "added charge other than service charge"
    },
    20: {
        "category": "sub_total.tax_price",
        "describe": "tax amount"
    },
    21: {
        "category": "sub_total.etc",
        "describe": "others"
    },
    22: {
        "category": "total.total_price",
        "describe": "total price"
    },
    23: {
        "category": "total.total_etc",
        "describe": "others"
    },
    24: {
        "category": "total.cashprice",
        "describe": "amount of price paid in cash"
    },
    25: {
        "category": "total.changeprice",
        "describe": "amount of change in cash"
    },
    26: {
        "category": "total.creditcardprice",
        "describe": "amount of price paid in credit/debit card"
    },
    27: {
        "category": "total.emoneyprice",
        "describe": "amount of price paid in emoney, point"
    },
    28: {
        "category": "total.menutype_cnt",
        "describe": "total count of type of menu"
    },
    29: {
        "category": "total.menuqty_cnt",
        "describe": "total count of quantity"
    }
}
ner_cls_map_category={value['category']: {"describe": value['describe'], "index": key} for key, value in NER_cls_map_id.items()}
dataset_name='CORD'
def convert_bbox(words):
    """
    将给定的多个文本框的 quad 转换为最小包围框 bbox 格式。
    """
    x_min = min(word['quad']['x1'] for word in words)
    y_min = min(word['quad']['y1'] for word in words)
    x_max = max(word['quad']['x2'] for word in words)
    y_max = max(word['quad']['y3'] for word in words)
    return [x_min, y_min, x_max, y_max]

def process_ocr(data):
    """
    处理 OCR 部分，将 words 转换为 OCR 格式。
    """
    ocr_list = []
    i=0
    for line in data["valid_line"]:
        for word in line["words"]:
            quad=word['quad']
            ocr_entry = {
                "id": i,
                "bbox": [quad['x1'],quad['y1'],quad['x2'],quad['y2'],quad['x3'],quad['y3'],quad['x4'],quad['y4']],
                "text": word["text"]
            }
            ocr_list.append(ocr_entry)
            i+=1
    return ocr_list
def find_ocr_id(quad,ocr_list):
    bbox=[quad['x1'],quad['y1'],quad['x2'],quad['y2'],quad['x3'],quad['y3'],quad['x4'],quad['y4']]
    for item in ocr_list:
        if item['bbox']==bbox:
            return item["id"]
def process_NER(data,ocr_list):
    """
    处理 KIE 部分，将 valid_line 转换为 NER 。
    """
    ner_list = []
    relation_list = []
    i=0
    for line in data["valid_line"]:
        ner_entry = {
            "id": line["group_id"],
            "ocr_info": {
                "all": {
                    "bbox": convert_bbox(line["words"]),  # 此处假设每个 group 只有一个 word，需根据实际情况调整
                    "text": " ".join([word["text"] for word in line["words"]])
                },
                "words": [find_ocr_id(word["quad"],ocr_list) for word in line["words"]]
            },
            "category": ner_cls_map_category[line["category"]]['index']
        }
        ner_list.append(ner_entry)
    return ner_list

def process_relation(data):
    """
    处理 KIE 部分，将 valid_line 转换为 relation 。
    """
    return []

def process_vqa():
    """
    处理 VQA 部分，返回空列表。
    """
    return []

def process_layout():
    """
    处理 Layout 部分，返回空列表。
    """
    return []

def process_caption():
    """
    处理 Caption 部分，返回空列表。
    """
    return []

def process_dense_description():
    """
    处理 Dense Description 部分，返回空对象。
    """
    return {}

def convert_to_unified_format(input_root, output_path,img_root):
    """
    读取输入文件并将其转换为统一数据集格式。
    """
    input_names=[os.path.join(input_root,name) for name in os.listdir(input_root)]

    unified_data = {
        "meta_info":{
            "dataset_name": dataset_name,
            "img_root":img_root
        },
        
        "data": [],
        "meta_dicts": {
            "NER_cls_map": NER_cls_map_id,
            "vqa_cls_map": {},
            "img_cls_map": {},
            "layout_cls_map": {}
        }
    }
    for i,input_path in enumerate(input_names):
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ocr_list=process_ocr(data)
        img_path=os.path.join(img_root,os.path.basename(input_path)[:-4]+'png')
        unified_data['data'].append(add_img_data(data,ocr_list,img_path).copy())
        # if i>10:
        #     break
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unified_data, f, ensure_ascii=False, indent=4)
def add_img_data(data,ocr_list,img_path):
    res={
        "img_meta":{
            "img_path": img_path,
            "shape":cv2.imread(os.path.join(dataset_name,img_path)).shape
        },
        
        "info": {
            "ocr": {
                "word_level":ocr_list,
                "line_level":[]
            },
            "kie": {
                "NER": process_NER(data,ocr_list),
                "relation": process_relation(data)
            },
            "vqa": process_vqa(),
            "layout": process_layout(),
            "caption": process_caption(),
            "dense_description": process_dense_description()
        }
    }
    return res
# 定义输入和输出文件路径
input_file_path = Path("CORD/train/json")
img_root='train/image'
output_file_path = Path("CORD/train/cord_train.json")

# 执行转换
convert_to_unified_format(input_file_path, output_file_path,img_root)
