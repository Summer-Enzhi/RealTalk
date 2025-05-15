from datasets import load_dataset, Audio
from argparse import ArgumentParser
from src.models import model_cls_mapping
import json
from tqdm import tqdm
from loguru import logger
from pydub import AudioSegment
import tempfile
import os
import re
import torch
import random
random.seed(42)


torch.cuda.device_count()
def process_segment(main_audio, start_ms, end_ms):
    os.makedirs("./tmp", exist_ok=True)
    
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav", 
        dir="./tmp",
        delete=False
    )
    segment = main_audio[start_ms:end_ms]
    segment.export(temp_file.name, format="wav")
    return temp_file.name

def extract_intent(response):
    response = response.lower().strip()
    patterns = [
        r"意图[：是]\s*([^$\（$\）\.。，,;；]+)",
        r"intent:\s*([^$\（$\）\.。，,;；]+)",
        r"意图是\s*'?\"?\s*([^'\"]+)",
        r"意图为\s*'?\"?\s*([^'\"]+)", 
        r"^([\u4e00-\u9fa5]+)[\(\（]",
        r"([\u4e00-\u9fa5]+)[。\.]*$"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            intent = match.group(1)
            intent = re.sub(r"[^\u4e00-\u9fa5]", "", intent)
            return intent.upper() if intent else None
    return None



def extract_slot(response, slot_name):
    if not response or not slot_name:
        return []
    
    response = response.strip()
    slot_name = slot_name.strip()
    
    patterns = [
        rf"{re.escape(slot_name)}是\s*[：:]\s*[$$【]([^$$】]+)[\]】]",
        rf"{re.escape(slot_name)}是\s*[：:]\s*([^\n]+)",
        rf"{re.escape(slot_name)}是\s*([^，,;；\s]+)",
    ]
    
    slot_value = None
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            slot_value = match.group(1).strip()
            slot_value = re.sub(r"['\"]", "", slot_value)
            slot_value = slot_value.replace("[","")
            slot_value = slot_value.replace("]","")
            break
    
    if not slot_value:
        return []
    
    separators = r",|，|;|；|、|\s+"
    split_values = re.split(separators, slot_value)
    
    result = [v.strip() for v in split_values if v.strip()]
    return result if result else []

def clean_text(context_text):
    return context_text.replace("～","").replace("⁇","").replace("✎","").replace("↻","")

def main():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen2', choices=list(model_cls_mapping.keys()))
    parser.add_argument('--data_basedir', type=str, default='')
    parser.add_argument('--data', type=str, default='single_domain_colloquial')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--modality', type=str, default='fix', choices=['audio', 'text', 'ttft', 'fix'])
    parser.add_argument('--task', type=str, default='IC')
    args = parser.parse_args()


    with open(args.data, 'r') as file:
        data = json.load(file)
    model = model_cls_mapping[args.model]()

    if args.modality == 'ttft':
        _ = model.generate_ttft(data[0]['audio'])

    # inference
    results = []
    for item in tqdm(data, total=len(data)):
        tmp = {k: v for k, v in item.items() if k != 'audio'}
        conversation_history = []
        context = item['original_data']['context']
        audio_file = os.path.join(args.data_basedir, item['audio_file']) 
        main_audio = AudioSegment.from_file(audio_file)

        if args.task == 'IC':
            
            conversation_history.append({
                'role':"system",
                'content':[{
                    'type':'text',
                    'text':"你是一名音频分析助理。你的任务是分析本次对话中最后一段音频背后的意图。"
                }]
            })
            
            if context != []:
                start = 0
                end = int(float(context[-1]['hdTimeEnd']) * 1000)
                audio_file_his = process_segment(main_audio, start, end)
                contents_his = [("audio",audio_file_his), ("text","以上是对话历史上下文。")]
                
                for type, content in contents_his:
                    conversation_history.append({
                        'role':"user",
                        'content':[{
                            'type':type,
                            type:content
                        }]
                    })
            
            choices = item['original_data']['choices']
            instruct = f"以上是最后一段的当前音频。仅仅从以下选项中选择一个最合适的答案作为最后一轮语音的意图(即目的或目标):{choices}。你的回答应该严格遵循以下格式： 意图是：xx"
           
            start = int(float(item['original_data']['hdTimeStart']) * 1000)
            end = int(float(item['original_data']['hdTimeEnd']) * 1000)
            audio_file_now = process_segment(main_audio, start, end)

            contents = [("audio",audio_file_now), ("text",instruct)]
            for type, content in contents:
                conversation_history.append({
                    'role':"user",
                    'content':[{
                        'type':type,
                        type:content
                    }]
                })
            error = False
            try:
                intent_pre = model.generate_fix(conversation_history, max_new_tokens = 256)
                intent_pre = extract_intent(intent_pre)
            except Exception as e:
                print(f"Error in intent extraction: {e}")
                error = True
                conversation_history = [conversation_history[0], conversation_history[-2], conversation_history[-1]]
                try:
                    intent_pre = model.generate_fix(conversation_history, max_new_tokens = 4096)
                    intent_pre = extract_intent(intent_pre)
                except:
                    print("Error in intent extraction again")
                    intent_pre = ""
            tmp['意图'] = {"预测意图":intent_pre, "实际意图":item['text'],"prompt":f"音频内容：{clean_text(item['original_data']['文本content'])}。{instruct}"}
        
        
        
            # SF任务
            conversation_history = []
            conversation_history.append({
                'role':"system",
                'content':[{
                    'type':'text',
                    'text':"你是一名音频分析助理。你的任务是分析本次对话中最后一段音频的相关关键信息。"
                }]
            })
            
            if context != [] and not error:
                contents_his = [("audio",audio_file_his), ("text","以上是对话历史上下文。")]
                
                for type, content in contents_his:
                    conversation_history.append({
                        'role':"user",
                        'content':[{
                            'type':type,
                            type:content
                        }]
                    })
            
            slots = item['original_data']['槽值字典']
            res = {}
            start = int(float(item['original_data']['hdTimeStart']) * 1000)
            end = int(float(item['original_data']['hdTimeEnd']) * 1000)
            audio_file_now = process_segment(main_audio, start, end)

            for k,v in slots.items():
                instruct = f"请仔细分析当前音频内容，并结合用户意图「{intent_pre}」提取关键信息。\n其中，「{k}」是该意图的重要属性，请从语音中识别仅仅与意图「{intent_pre}」相关的「{k}」的具体内容。\n回答时请严格遵循以下格式：\n{k}是：[具体值1,具体值2,...具体值n]"
                
                contents = [("audio",audio_file_now), ("text",instruct)]
                for type, content in contents:
                    conversation_history.append({
                        'role':"user",
                        'content':[{
                            'type':type,
                            type:content
                        }]
                    })
                try:
                    pre_v = model.generate_fix(conversation_history, max_new_tokens = 256)
                    conversation_history = conversation_history[:-2]
                    pre_v_list = extract_slot(pre_v,slot_name=k)
                    pre_v_list = list(set(pre_v_list))
                    pre_v_list.sort()
                    v.sort()
                except Exception as e:
                    print(f"Error in slot extraction: {e}")
                    pre_v_list = []
                    conversation_history = conversation_history[:-2]
                res[k] = {"预测值":pre_v_list, "实际值":v, "prompt":f"音频内容：{clean_text(item['original_data']['文本content'])}。{instruct}"}
            
            tmp['槽填充'] = res
            logger.info(f"\n意图：{tmp['意图']}")
            logger.info(f"\n槽填充：\n{json.dumps(tmp['槽填充'],ensure_ascii=False,indent=2)}")
            logger.info('====================================')
            results.append(tmp)
            
            
        # roleID 1 : user
        elif args.task == 'chat' and item['original_data']['roleID'] == 1 and (item['next']=={} or item['next']['roleID'] == 2):
            conversation_history.append({
                'role':"system",
                'content':[{
                    'type':'text',
                    'text':"你是一名语音对话助理。你的任务是回答用户相关问题。"
                }]
            })
            
            context_text = ""
            for single in context:
                start = int(float(single['hdTimeStart']) * 1000)
                end = int(float(single['hdTimeEnd']) * 1000)
                audio_file_his = process_segment(main_audio, start, end)
                if single['roleID'] == 1:
                    role = 'user'
                else:
                    role = 'assistant'
                    
                if args.model == 'gpt4o_mini' and role == 'assistant':
                    conversation_history.append({
                        'role': role,
                        'content':[{
                            'type':"text",
                            "text":clean_text(single['text'])
                        }]
                    })
                else:
                    conversation_history.append({
                        'role': role,
                        'content':[{
                            'type':"audio",
                            "audio":audio_file_his
                        }]
                    })
                context_text += f"{role}: {single['text']}\n"
            
            start = int(float(item['original_data']['hdTimeStart']) * 1000)
            end = int(float(item['original_data']['hdTimeEnd']) * 1000)
            audio_file_now = process_segment(main_audio, start, end)
            conversation_history.append({
                'role':"user",
                'content':[{
                    'type':'audio',
                    'audio':audio_file_now
                }]
            })
            context_text += f"user: {item['original_data']['文本content']}\n"
            
            context_text = clean_text(context_text)
            try:
                response = clean_text(model.generate_fix(conversation_history, max_new_tokens = 256))
            except Exception as e:
                print(f"Error in chat response generation: {e}")
                # raise e
                response = ""
                
            tmp['对话'] = {
                "预测回复":response, 
                "实际回复":clean_text(item['next']['text']) if item['next'] != {} else "",
                "prompt":f"对话上文：{context_text}，此时assistant的回复应该是什么？"
                }

            logger.info(f"\对话: \n{json.dumps(tmp['对话'],ensure_ascii=False,indent=2)}")
            logger.info('====================================')
            results.append(tmp)
        
        

        elif args.task == 'multimodality_chat' and item['original_data']['roleID'] == 1 and (item['next']=={} or item['next']['roleID'] == 2):
            
            conversation_history.append({
                'role':"system",
                'content':[{
                    'type':'text',
                    'text':"你是一名多模态对话助理。你的任务是处理用户通过语音或文本提出的问题。"
                }]
            })
            
            context_text = ""
            idx = 0
            for single in context:
                role = 'user' if single['roleID'] == 1 else 'assistant'

                flag = False
                for mark in ['～', '⁇', '✎', '↻']:
                    if mark in single['text']:
                        flag = True
                        break
                if flag:
                    use_audio = True
                else:

                    use_audio = False   
                    
                if args.model == 'gpt4o_mini' and role == 'assistant':
                    use_audio = False
                          
                content = []
                if use_audio and 'hdTimeStart' in single and 'hdTimeEnd' in single:
                    start = int(float(single['hdTimeStart']) * 1000)
                    end = int(float(single['hdTimeEnd']) * 1000)
                    audio_file_his = process_segment(main_audio, start, end)
                    content.append({
                        'type': 'audio',
                        'audio': audio_file_his
                    })
                    print(f'Turn {idx}: Audio')
                else:
                    content.append({
                        'type': 'text',
                        'text': single.get('text', '')
                    })

                    print(f'Turn {idx}: Text')
                    
                conversation_history.append({
                    'role': role,
                    'content': content
                })
                
                context_text += f"{role}: {single.get('text', '')}\n"
            

            current_data = item['original_data']
            use_current_audio = True 
            
            current_content = []
            if use_current_audio and 'hdTimeStart' in current_data and 'hdTimeEnd' in current_data:
                start = int(float(current_data['hdTimeStart']) * 1000)
                end = int(float(current_data['hdTimeEnd']) * 1000)
                audio_file_now = process_segment(main_audio, start, end)
                current_content.append({
                    'type': 'audio',
                    'audio': audio_file_now
                })
            else:
                current_content.append({
                    'type': 'text',
                    'text': clean_text(current_data.get('文本content', ''))
                })
            
            conversation_history.append({
                'role': 'user',
                'content': current_content
            })
            context_text += f"user: {current_data.get('文本content', '')}\n"
            
            context_text = clean_text(context_text)
            try:
                response = clean_text(model.generate_fix(conversation_history, max_new_tokens=256))
            except Exception as e:
                print(f"Error in multimodal response generation: {e}")
                response = ""
                
            tmp['混合模态对话'] = {
                "用户模态": "语音" if use_current_audio else "文本",
                "预测回复": response,
                "实际回复": clean_text(item['next']['text']) if item['next'] != {} else "",
                "prompt": f"对话上文：{context_text}，此时assistant的回复应该是什么？"
            }

            logger.info(f"\n混合模态对话: \n{json.dumps(tmp['混合模态对话'], ensure_ascii=False, indent=2)}")
            logger.info('====================================')
            results.append(tmp)

    output_dir = f'results/{args.model}/{args.task}'
    output_file = f'{output_dir}/{args.data.split("/")[-2]}-{args.split}-{args.modality}.json'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    os.system(f"python evaluate.py --src_file {output_file} --evaluator {args.task}")  

if __name__ == '__main__':
    main()
