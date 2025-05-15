from .base import VoiceAssistant
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import soundfile as sf
import numpy as np
from resampy import resample
import torch

class Qwen2Assistant(VoiceAssistant):
    def __init__(self):
        self.path = "Qwen/Qwen2-Audio-7B-Instruct"
        
        # 多卡并行配置
        self.processor = AutoProcessor.from_pretrained(self.path, cache_dir='./cache')
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.path,
            device_map="auto",
            cache_dir='./cache',
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            offload_folder="./offload"
        ).eval()

    def _prepare_inputs(self, inputs):
        return {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()}

    def generate_audio(self, audio, max_new_tokens=2048):
        assert audio['sampling_rate'] == 16000
        
        processed_audio = audio['array']
        content = [{"type": "audio", "audio_url": 'xxx'}]
        conversation = [{"role": "user", "content": content}]
        
        inputs = self.processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True,
            tokenize=False
        )
        processed_inputs = self.processor(
            text=inputs,
            audios=[processed_audio],
            return_tensors="pt",
            padding=True
        )
        processed_inputs = self._prepare_inputs(processed_inputs)
        
        generate_ids = self.model.generate(
            **processed_inputs,
            max_new_tokens=max_new_tokens
        )
        generate_ids = generate_ids[:, processed_inputs.input_ids.size(1):]
        
        return self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

    def generate_text(self, text):
        content = [{"type": "text", "text": text}]
        conversation = [{"role": "user", "content": content}]
        
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        processed_inputs = self.processor(
            text=inputs,
            audios=None,
            return_tensors="pt",
            padding=True
        )
        processed_inputs = self._prepare_inputs(processed_inputs)
        
        generate_ids = self.model.generate(
            **processed_inputs,
            max_length=2048,
            do_sample=True,
            temperature=0.7
        )
        return self.processor.batch_decode(
            generate_ids[:, processed_inputs.input_ids.size(1):],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

    def _load_audio(self, file_path):
        try:
            audio, sr = sf.read(file_path, dtype='float32', always_2d=True)
            audio = np.mean(audio, axis=1)
            if sr != 16000:
                audio = resample(audio, sr, 16000)
            return audio
        except Exception as e:
            raise ValueError(f"音频加载失败: {file_path} - {str(e)}")

    def generate_fix(self, conversation_history, max_new_tokens=2048):
        audios = []
        for msg in conversation_history:
            if msg['content'][0]['type'] == 'audio':
                audio = self._load_audio(msg['content'][0]['audio'])
                audios.append(audio)
        
        text_inputs = self.processor.apply_chat_template(
            conversation_history,
            add_generation_prompt=True,
            tokenize=False
        )
        processed_inputs = self.processor(
            text=text_inputs,
            audios=audios if audios else None,
            return_tensors="pt",
            padding=True,
            sampling_rate=16000
        )
        processed_inputs = self._prepare_inputs(processed_inputs)
        
        generated_ids = self.model.generate(
            **processed_inputs,
            max_new_tokens=max_new_tokens        )
        
        return self.processor.batch_decode(
            generated_ids[:, processed_inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]