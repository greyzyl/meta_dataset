import json
import os
import requests

def download_image(image_url, save_path, bad_urls):
    try:
        if not os.path.exists(save_path):
            response = requests.get(image_url)
            response.raise_for_status()  # 检查请求是否成功
            with open(save_path, 'wb') as file:
                file.write(response.content)
            print(f"图片成功保存到 {save_path}")
        else:
            print(f"{save_path}已下载")
        return []
    except requests.exceptions.RequestException as e:
        print(f"下载图片时出错: {e}")
        with open('TextCaps/bad_url.json','w',encoding='utf-8') as f:
            json.dump(bad_urls,f)
        return [image_url]

def download_images2dir(image_metas,save_root):
    bad_urls=[]
    for image_meta in image_metas:
        image_name=image_meta['image_name']
        image_url=image_meta['url']
        save_path=os.path.join(save_root,image_name)
        bad_url=download_image(image_url,save_path,bad_urls)
        bad_urls+=bad_url
    return bad_urls

with open('TextCaps/caption_json/TextCaps_0.1_train.json','r',encoding='utf-8')as f:
    js=json.load(f)
image_urls=[]
for idx,image_item in enumerate(js['data']):
    image_urls.append({
        'url':image_item['flickr_original_url'],
        'image_name':os.path.basename(image_item['image_path'])
        }) 
    # if idx>10:
    #     break
os.makedirs('TextCaps/download_ori_img',exist_ok=True)
bad_urls=download_images2dir(image_urls,'TextCaps/download_ori_img')
with open('TextCaps/bad_url.json','w',encoding='utf-8') as f:
    json.dump(bad_urls,f)
