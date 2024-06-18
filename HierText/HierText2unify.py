import json
import os
from pathlib import Path

import cv2
dataset_name='DocVQA'
def convert_bbox(words):
    """
    将给定的多个文本框的 boundingBox 转换为最小包围框 bbox 格式。
    """
    x_min = min(word['boundingBox'][0] for word in words)
    y_min = min(word['boundingBox'][1] for word in words)
    x_max = max(word['boundingBox'][4] for word in words)
    y_max = max(word['boundingBox'][5] for word in words)
    return [x_min, y_min, x_max, y_max]

def process_ocr_word(data):
    """
    处理 OCR 部分，将 words 转换为 OCR 格式。
    """
    ocr_list = []
    i=0
    for annotation in data['annotations']:
        for paragragh in annotation['paragraphs']:
            for line in paragragh["lines"]:
                for word in paragragh["words"]:
                    boundingBox=word['boundingBox']
                    boundingBox=[boundingBox[0],boundingBox[1],boundingBox[2],boundingBox[3],boundingBox[4],boundingBox[5],boundingBox[6],boundingBox[7]]
                    text='###'
                    if word['legible']:
                        text=word["text"]
                    ocr_entry = {
                        "id": i,
                        "bbox": boundingBox,
                        "text": text
                    }
                    ocr_list.append(ocr_entry)
                    i+=1
    return ocr_list
def process_ocr_line(data,ocr_list):
    line_list=[]
    i=0
    for annotation in data['annotations']:
        for paragragh in annotation['paragraphs']:
            for line in paragragh["lines"]:
                for word in paragragh["words"]:
                    boundingBox=word['boundingBox']
                    boundingBox=[boundingBox[0],boundingBox[1],boundingBox[2],boundingBox[3],boundingBox[4],boundingBox[5],boundingBox[6],boundingBox[7]]
                    text='###'
                    if word['legible']:
                        text=word["text"]
                    ocr_entry = {
                        "id": i,
                        "bbox": boundingBox,
                        "text": text
                    }
                    ocr_list.append(ocr_entry)
                    i+=1
    return ocr_list
    return line_list

def find_ocr_id(boundingBox,ocr_list):
    bbox=[boundingBox[0],boundingBox[1],boundingBox[2],boundingBox[3],boundingBox[4],boundingBox[5],boundingBox[6],boundingBox[7]]
    for item in ocr_list:
        if item['bbox']==bbox:
            return item["id"]
        
def process_NER(data,ocr_list):
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

def process_relation(data):
    """
    处理 KIE 部分，将 valid_line 转换为 relation 。
    """
    return []

def process_vqa(vqa_data,img_name):
    """
    处理 VQA 部分，将 DocVQA 数据集格式转换为统一数据集格式。
    """
    vqa_list = []
    question_id=0
    answer_id=0
    for entry in vqa_data["data"]:
        if os.path.basename(entry['image'])==img_name:
            vqa_entry = {
                "question_id": question_id,
                "question": entry["question"],
                "answers": [],
                "img_clses": []  # 使用question_types作为img_clses
            }
            question_id+=1
            answers=[]
            for idx, answer in enumerate(entry["answers"]):
                answers.append({
                        "answer_id": answer_id,
                        "answer": answer,
                        "confidence": "yes",  # DocVQA数据集中没有提供confidence，默认设为yes
                        "evidence": "",
                        "answer_source": entry['question_types'],
                        "operation": ""
                    })
            answer_id+=1
            vqa_entry["answers"]=answers
            vqa_list.append(vqa_entry)
    return vqa_list

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
            "img_path": img_path,
            "shape":cv2.imread(os.path.join(img_path)).shape
        },
        
        "info": {
            "ocr": {
                "word_level":OCR_list,
                "line_level":process_ocr_line(OCR_data,OCR_list)
            },
            "kie": {
                "NER": process_NER(KIE_NER_data,OCR_list),
                "relation": process_relation(KIE_relation_data)
            },
            "vqa": process_vqa(VQA_data,os.path.basename(img_path)),
            "layout": process_layout(),
            "caption": process_caption(),
            "dense_description": process_dense_description()
        }
    }
    return res

def convert_to_unified_format(ocr_root,vqa_path, output_path,img_root):
    """
    读取输入文件并将其转换为统一数据集格式。
    """
    ocr_json_names=[os.path.join(ocr_root,name) for name in os.listdir(ocr_root)]
    with open(vqa_path,'r',encoding='utf-8')as f:
        vqa_data=json.load(f)
    img_names=[os.path.basename(item['image']) for item in vqa_data['data']]
    print(len(img_names))
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
    for i,ocr_json_name in enumerate(ocr_json_names):
        bad_case=[]
        try:
            with open(ocr_json_name, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
            ocr_list=process_ocr_word(ocr_data)
            if os.path.basename(ocr_json_name)[:-4]+'png' in img_names:
                img_path=os.path.join(img_root,os.path.basename(ocr_json_name)[:-4]+'png')
                unified_data['data'].append(
                    add_img_data(img_path,OCR_list=ocr_list,OCR_data=ocr_data,VQA_data=vqa_data).copy()
                    )
            
        except:
            bad_case.append(ocr_json_name)
        print(i)
        # if i>-1:
        #     break
    print('bad_case',bad_case)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unified_data, f, ensure_ascii=False, indent=4)

# 定义输入和输出文件路径
ocr_file_root = Path("DocVQA/ocr_json")
vqa_file_path=Path("DocVQA/vqa_json/train_v1.0_withQT.json")
img_root='DocVQA/image'
output_file_path = Path("DocVQA/DocVQA_train.json")

# 执行转换
convert_to_unified_format(ocr_file_root,vqa_file_path, output_file_path,img_root)
