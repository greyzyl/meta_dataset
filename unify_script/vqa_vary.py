import json


json_path='/home/yuhaiyang/cyw/TextVQA/example.json'
img_root=''
sys_prompt='Answer the question using a single number or phrase.'
save_path='TextVQA_train_vary.json'
def process_vqa(data,i,sys_prompt=''):
    conversations=[]
    question=data['info']['vqa'][i]
    question_conversation={
        'from':'human',
        'value':''
    }
    answer_conversation={
        'from':'gpt',
        'value':''
    }
    question_str='<image>\n'+question['question']+'\n'+sys_prompt
    question_conversation['value']=question_str
    #找到第一个置信度为yes的answer
    for answer in question['answers']:
        if answer['confidence']=='yes':
            break
    answer_str=answer['answer']
    answer_conversation['value']=answer_str
    conversations.append(question_conversation)
    conversations.append(answer_conversation)
    return conversations

def process_pre_img(data,sys_prompt=''):
    res_pre_img=[]
    question_num=len(data['info']['vqa'])
    for i in range(question_num):#每一个问题和它的第一个回答构成一个conversation
        res={
            'image':data['img_meta']['img_path'],
            'conversations':process_vqa(data,i,sys_prompt)
            }
        res_pre_img.append(res)
    return res_pre_img
    
def process_dataset(json_path,save_path,sys_prompt=''):
    res_all=[]
    with open(json_path,'r',encoding='utf-8') as f:
        all_data=json.load(f)
    for data in all_data['data']:
        res_pre_img=process_pre_img(data,sys_prompt)
        res_all+=res_pre_img
    with open(save_path,'w',encoding='utf-8') as f:
        json.dump(res_all,f,ensure_ascii=False,indent=4)
process_dataset(json_path,save_path,sys_prompt)
    