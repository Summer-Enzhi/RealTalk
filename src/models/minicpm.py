from .base import VoiceAssistant
from transformers import AutoModel, AutoTokenizer
import transformers
import torch
import soundfile as sf
from resampy import resample
import numpy as np
import librosa

class MiniCPMAssistant(VoiceAssistant):
    def __init__(self):
        self.model = AutoModel.from_pretrained(
            'openbmb/MiniCPM-o-2_6',
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16,
            device_map="auto",  # 自动分配多卡
            init_vision=False,
            init_audio=True,
            init_tts=False
        )
        self.model = self.model.eval()  # 移除.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            'openbmb/MiniCPM-o-2_6', 
            trust_remote_code=True
        )

        self.sys_prompt = self.model.get_sys_prompt(mode='audio_assistant', language='en')
        print(self.sys_prompt)

    def _prepare_audio_input(self, audio_array):
        audio_tensor = torch.from_numpy(audio_array).to(self.model.device)
        return audio_tensor.float()  # 确保float32类型

    def generate_audio(
        self,
        audio,
        max_new_tokens=2048,
    ):
        processed_audio = self._prepare_audio_input(audio['array'])
        user_question = {'role': 'user', 'content': [processed_audio]}
        msgs = [self.sys_prompt, user_question]
        
        res = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True,
            max_new_tokens=max_new_tokens,
            use_tts_template=True,
            generate_audio=False,
            temperature=0.3,
        )
        return res

    def _load_audio(self, file_path):
        try:
            audio, sr = sf.read(file_path, dtype='float32', always_2d=True)
            audio = np.mean(audio, axis=1)
            
            if sr != 16000:
                audio = resample(audio, sr, 16000)
                sr = 16000
                
            return audio
        except Exception as e:
            raise ValueError(f"音频文件加载失败: {file_path} - {str(e)}")

    def generate_fix(
        self,
        conversation_history,
        max_new_tokens=2048,
    ):
        his = []
        for message in conversation_history:
            if message['content'][0]['type'] == 'audio':
                audio_file = message['content'][0]['audio']
                processed_audio = librosa.load(audio_file, sr=16000, mono=True)[0]
                his.append({"role":message['role'],
                            "content":[processed_audio]})
            else:
                his.append({"role":message['role'],
                            "content":[message['content'][0]['text']]})
        res = self.model.chat(
            msgs=his,
            tokenizer=self.tokenizer,
            sampling=False,
            max_new_tokens=max_new_tokens,
            use_tts_template=True,
            generate_audio=False,
            temperature=0.3,
        )
        return res