from .base import Evaluator
import numpy as np
from qa_metrics.pedant import PEDANT
import jiwer
from tqdm import tqdm

import string
chinese_punctuation = "，。！？；：“”‘’（）【】…—～"
all_punctuation = string.punctuation + chinese_punctuation

def remove_trailing_punctuation(text):
    return text.rstrip(all_punctuation)

class ICEvaluator(Evaluator):
    def __init__(self):
        self.pedant = PEDANT()
    
    def evaluate(self, data):
        res = []
        # Intent Recognition Metrics
        intent_correct = 0
        
        # Slot Filling Metrics
        slot_accuracy, slot_precision, slot_recall, slot_f1 = 0, 0, 0, 0
        slot_wer, slot_panda = [], []
        total_slots = 0
        intent_pandas = []
        
        total_jga_items = 0
        correct_jga_items = 0

        for item in tqdm(data):
            # Evaluate Intent
            pred_intent = item['意图']['预测意图']
            true_intent = item['意图']['实际意图']
            prompt = item['意图']['prompt']
            intent_now = -1
            intent_panda_now = -1
            if pred_intent != None and true_intent != None:
                intent_now = 0
                if pred_intent == true_intent:
                    intent_correct += 1
                    intent_now = 1
                    
                intent_panda = self.pedant.evaluate(
                            [true_intent.lower()],
                            pred_intent.lower(),
                            prompt.lower()
                        )
                intent_panda_now = int(intent_panda)
                intent_pandas.append(intent_panda)
            
            # Evaluate Slots
            item_f1 = 0
            item_panda = 0
            all_slots_correct = True  # 初始假设所有槽位都正确
            if '槽填充' in item:
                for slot_name, slot_info in item['槽填充'].items():
                    total_slots += 1
                    pred_values = [remove_trailing_punctuation(i) for i in slot_info['预测值']]
                    item['槽填充'][slot_name]['预测值'] = pred_values
                    true_values = slot_info['实际值']
                    prompt = slot_info['prompt']
                    
                    # 1. Accuracy (Exact Match)
                    if pred_values == true_values:
                        slot_accuracy += 1
                    else:
                        all_slots_correct = False  # 如果有任何一个槽位不匹配，标记为不完全正确
                    
                    # 如果有任何一个槽位不匹配，标记为不完全正确
            
                    # 2. Precision/Recall/F1
                    pred_set = set(pred_values)
                    true_set = set(true_values)
                    tp = len(pred_set & true_set)
                    fp = len(pred_set - true_set)
                    fn = len(true_set - pred_set)
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    slot_precision += precision
                    slot_recall += recall
                    slot_f1 += f1
                    item_f1 += f1
                    
                    # 3. WER (Convert to string)
                    miniLength = min(len(pred_values), len(true_values))
                    pred_str = ' '.join(pred_values[:miniLength])
                    true_str = ' '.join(true_values[:miniLength])
                    if pred_str != "" and true_str != "":
                        slot_wer.append(jiwer.wer(true_str, pred_str))
                    
                        
                        # 4. PandaScore
                        panda_score = self.pedant.evaluate(
                            [true_str.lower()],
                            pred_str.lower(),
                            prompt.lower()
                        )
                        item_panda += int(panda_score)
                        slot_panda.append(panda_score)
                         
            if item['槽填充'] == {}:
                item_jga = -1
            elif all_slots_correct == True:
                item_jga = 1
                correct_jga_items += 1
                total_jga_items += 1    
            else:
                item_jga = 0
                total_jga_items += 1    
            # Calculate Metrics
            tmp = {k: v for k, v in item.items() if k != 'audio'}
            tmp['result'] = {}
            tmp['result']['IC_Accuracy'] = intent_now
            tmp['result']['IC_PANDA'] =  intent_panda_now
            tmp['result']['SF_F1'] = item_f1 / len(item['槽填充']) if item['槽填充'] != {} else -1
            tmp['result']['SF_PANDA'] =  item_panda / len(item['槽填充']) if item['槽填充'] != {} else -1
            tmp['result']['SF_JGA'] =  item_jga
            res.append(tmp)
        
        metrics = {}
        metrics['intent_metric'] = {
                'accuracy': (intent_correct / len(data)) * 100 if data else 0,
                'intent_panda': (np.mean(intent_pandas) if intent_pandas else 0) * 100
            }
        
        
        if total_slots > 0:
            metrics['slot_metric'] = {
                'accuracy': (slot_accuracy / total_slots) * 100,
                'precision': (slot_precision / total_slots) * 100,
                'recall': (slot_recall / total_slots) * 100,
                'f1': (slot_f1 / total_slots) * 100,
                'wer': (np.mean(slot_wer) * 100 if slot_wer else 0),
                'pandascore': (np.mean(slot_panda) if slot_panda else 0) * 100,
                'jga': (correct_jga_items / total_jga_items) * 100 if total_jga_items > 0 else 0
            }
        else:
            metrics['slot_metric'] = {}
        res.append(metrics)
        
        return res