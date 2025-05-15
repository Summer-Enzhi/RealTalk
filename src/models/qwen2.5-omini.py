from .base import VoiceAssistant
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import torch

class Qwen25OmniAssistant(VoiceAssistant):
    def __init__(self, flash_attention=False):
        self.model_path = "Qwen/Qwen2.5-Omni-7B"
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
        
        model_args = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto",
            "low_cpu_mem_usage": True, 
            "offload_folder": "./offload"
        }
        if flash_attention:
            if torch.cuda.get_device_capability()[0] >= 8:
                model_args["attn_implementation"] = "flash_attention_2"
            else:
                print("当前GPU不支持flash attention，已自动回退到sdpa")
        
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path,
            **model_args
        ).eval() 
        
        self.system_prompt = {
            "role": "system",
            "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group."}]
        }

    def _prepare_inputs(self, inputs):
        return {k: (v.to(self.model.device) if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()}

    def generate_text(self, text, max_new_tokens=2048):
        conversation = [
            self.system_prompt,
            {"role": "user", "content": [{"type": "text", "text": text}]}
        ]
        
        text_inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        audios, images, videos = process_mm_info(conversation)
        
        inputs = self.processor(
            text=text_inputs,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        )
        inputs = self._prepare_inputs(inputs)  # 统一设备分配
        
        text_ids, _ = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        return self.processor.batch_decode(
            text_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

    def generate_fix(self, conversation_history, max_new_tokens=2048):
        if all(msg["role"] != "system" for msg in conversation_history):
            conversation_history = [self.system_prompt] + conversation_history
        
        text_inputs = self.processor.apply_chat_template(
            conversation_history,
            add_generation_prompt=True,
            tokenize=False
        )
        audios, images, videos = process_mm_info(conversation_history)
        
        inputs = self.processor(
            text=text_inputs,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        )
        inputs = self._prepare_inputs(inputs)
        
        text_ids, _ = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.processor.tokenizer.pad_token_id
        )
        
        return self.processor.batch_decode(
            text_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]