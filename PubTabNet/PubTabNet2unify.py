import json
import os
from pathlib import Path

import cv2
import re
from bs4 import BeautifulSoup as bs
import jsonlines
dataset_name='PubTabNet'
def convert_bbox(words):
    """
    将给定的多个文本框的 boundingBox 转换为最小包围框 bbox 格式。
    """
    x_min = min(word['boundingBox'][0] for word in words)
    y_min = min(word['boundingBox'][1] for word in words)
    x_max = max(word['boundingBox'][4] for word in words)
    y_max = max(word['boundingBox'][5] for word in words)
    return [x_min, y_min, x_max, y_max]


def is_html_label(text):
    return len(re.findall(r"<.*?>", text))>0

def format_html(label_dict):
    ''' Formats HTML code from tokenized annotation of img
    '''
    html_string = '''<html>
                     <head>
                     <meta charset="UTF-8">
                     <style>
                     table, th, td {
                       border: 1px solid black;
                       font-size: 10px;
                     }
                     </style>
                     </head>
                     <body>
                     <table frame="hsides" rules="groups" width="100%%">
                         %s
                     </table>
                     </body>
                     </html>''' % ''.join(label_dict['html']['structure']['tokens'])
    cell_nodes = list(re.finditer(r'(<td[^<>]*>)(</td>)', html_string))
    assert len(cell_nodes) == len(label_dict['html']['cells']), 'Number of cells defined in tags does not match the length of cells'
    cells = [''.join(c['tokens']) for c in label_dict['html']['cells']]
    offset = 0
    for n, cell in zip(cell_nodes, cells):
        html_string = html_string[:n.end(1) + offset] + cell + html_string[n.start(2) + offset:]
        offset += len(cell)
    # prettify the html
    soup = bs(html_string)
    html_string = soup.prettify()
    return html_string

def process_ocr_word(data):
    """
    处理 OCR 部分，将 words 转换为 OCR 格式。
    """
    ocr_list = []
    i=0
    cells=data['html']['cells']
    for cell in cells:
        boundingBox=[]
        if 'bbox' in cell.keys():
            boundingBox=cell['bbox']
            boundingBox=[boundingBox[0],boundingBox[1],boundingBox[2],boundingBox[1],boundingBox[2],boundingBox[3],boundingBox[0],boundingBox[3]]
        tokens=[]
        for text in cell['tokens']:
            if not is_html_label(text):
                tokens.append(text)
        text=''.join(tokens)
        ocr_entry = {
            "id": i,
            "bbox": boundingBox,
            "text": text
        }
        ocr_list.append(ocr_entry)
        i+=1
    return ocr_list
def process_ocr_line():
    return []

def find_ocr_id(boundingBox,ocr_list):
    bbox=[boundingBox[0],boundingBox[1],boundingBox[2],boundingBox[3],boundingBox[4],boundingBox[5],boundingBox[6],boundingBox[7]]
    for item in ocr_list:
        if item['bbox']==bbox:
            return item["id"]
        
def process_NER():
    """
    处理 KIE 部分，将 valid_line 转换为 NER 。
    """
    # ner_list = []
    # relation_list = []
    # i=0
    # for line in data["valid_line"]:
    #     ner_entry = {
    #         "id": line["group_id"],
    #         "ocr_info": {
    #             "all": {
    #                 "bbox": convert_bbox(line["words"]),  # 此处假设每个 group 只有一个 word，需根据实际情况调整
    #                 "text": " ".join([word["text"] for word in line["words"]])
    #             },
    #             "words": [find_ocr_id(word["boundingBox"],ocr_list) for word in line["words"]]
    #         },
    #         "category": ner_cls_map_category[line["category"]]['index']
    #     }
    #     ner_list.append(ner_entry)
    # return ner_list
    return []

def process_relation():
    """
    处理 KIE 部分，将 valid_line 转换为 relation 。
    """
    return []

def process_vqa():
    """
    处理 VQA 部分，将 DocVQA 数据集格式转换为统一数据集格式。
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

def process_dense_description(dense_description_data):
    """
    处理 Dense Description 部分，返回空对象。
    """
    res={
        "xml_type":"html",
        "xml": format_html(dense_description_data)
    }
    return res

def add_img_data(img_path,
                 OCR_list=None,
                 OCR_data=None,
                 KIE_NER_data=None,KIE_relation_data=None,
                 VQA_data=None,
                 layout_data=None,
                 caption_data=None,
                 dense_description_data=None):
    res={
        "img_meta":{
            "img_path": os.path.basename(img_path),
            "shape":cv2.imread(os.path.join(img_path)).shape
        },
        
        "info": {
            "ocr": {
                "word_level":OCR_list,
                "line_level":process_ocr_line()
            },
            "kie": {
                "NER": process_NER(),
                "relation": process_relation()
            },
            "vqa": process_vqa(),
            "layout": process_layout(),
            "caption": process_caption(),
            "dense_description": process_dense_description(dense_description_data)
        }
    }
    return res

def convert_to_unified_format(vqa_path, output_path,img_root):
    """
    读取输入文件并将其转换为统一数据集格式。
    """
    # with open(vqa_path,'r',encoding='utf-8')as f:
    #     vqa_data=json.load(f)
    datas=[]
    unified_data = {
        "meta_info":{
            "dataset_name": dataset_name,
            "img_root":img_root
        },
        
        "data": [],
        "meta_dicts": {
            "NER_cls_map": {},
            "vqa_cls_map": {},
            "img_cls_map": {},
            "layout_cls_map": {}
        }
    }
    bad_case=[]
    with open(vqa_path, "r+", encoding="utf-8") as f:
        for i,img_item in enumerate(jsonlines.Reader(f)):
            try:
                print(i)
                ocr_list=process_ocr_word(img_item)
                img_path=os.path.join(img_root,img_item['filename'])
                unified_data['data'].append(
                    add_img_data(img_path,OCR_list=ocr_list,dense_description_data=img_item).copy()
                    )
            except:
                bad_case.append(img_item)
            # print(i)
            # if i>-1:
            #     break
    print('bad_case',bad_case)
    print(len(bad_case))
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unified_data, f, ensure_ascii=False, indent=4)

def analysis_bad_case(vqa_path, output_path,img_root,bad_case):
    """
    读取输入文件并将其转换为统一数据集格式。
    """
    # with open(vqa_path,'r',encoding='utf-8')as f:
    #     vqa_data=json.load(f)
    datas=[]
    unified_data = {
        "meta_info":{
            "dataset_name": dataset_name,
            "img_root":img_root
        },
        
        "data": [],
        "meta_dicts": {
            "NER_cls_map": {},
            "vqa_cls_map": {},
            "img_cls_map": {},
            "layout_cls_map": {}
        }
    }
    
    with open(vqa_path, "r+", encoding="utf-8") as f:
        for i,img_item in enumerate(jsonlines.Reader(f)):
            # if os.path.basename(img_item['filename']) in bad_case:
                # print(i)
            #     ocr_list=process_ocr_word(img_item)
            #     img_path=os.path.join(img_root,img_item['filename'])
            #     unified_data['data'].append(
            #         add_img_data(img_path,OCR_list=ocr_list,dense_description_data=img_item).copy()
            #         )

            print(i)
            # if i>-1:
            #     break
    print('bad_case',bad_case)
    print(len(bad_case))
    # with open(output_path, 'w', encoding='utf-8') as f:
    #     json.dump(unified_data, f, ensure_ascii=False, indent=4)
# bad_case=['PMC3166891_003_00.png']

# 定义输入和输出文件路径
dense_description_file_path=Path("PubTabNet/dense_description_json/PubTabNet_2.0.0.jsonl")
img_root='PubTabNet/image/train'
output_file_path = Path("PubTabNet/PubTabNet.json")

# 执行转换
convert_to_unified_format(dense_description_file_path, output_file_path,img_root)
