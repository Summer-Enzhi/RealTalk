from .base import VoiceAssistant
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from src.api.gpt import generate_text_chat
from openai import OpenAI
import soundfile as sf
import numpy as np
import resampy

import http.client
import json

API_SECRET_KEY = 'Your API_SECRET_KEY'
BASE_URL = 'Your BASE_URL'
client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)


def chat_completions3(query):
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=query
    )
    return resp.choices[0].message.content

class Naive2Assistant(VoiceAssistant):
    def __init__(self):
        self.asr = self.load_asr()
        self.asr_cache = {}

    def load_asr(self):
        model_id = "openai/whisper-large-v3"
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            use_safetensors=True,
            cache_dir='./cache'
        )

        processor = AutoProcessor.from_pretrained(model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16,
            device_map="auto",
            model_kwargs={"use_safetensors": True}
        )
        return pipe

    def _load_audio(self, file_path):
        try:
            audio, sr = sf.read(file_path, dtype='float32', always_2d=True)
            audio = np.mean(audio, axis=1)
            if sr != 16000:
                audio = resampy.resample(audio, sr, 16000)
            return audio
        except Exception as e:
            raise ValueError(f"音频加载失败: {file_path} - {str(e)}")

    def generate_response(self, input_data, is_audio=True, max_tokens=2048):
        if is_audio:
            audio = self._load_audio(input_data)
            if input_data in self.asr_cache:
                transcript = self.asr_cache[input_data]
            else:
                transcript = self.asr(
                    audio, 
                    generate_kwargs={
                        "language": "chinese",
                        "return_timestamps": False                }
                )['text'].strip()
                self.asr_cache[input_data] = transcript 
            messages = [{"role": "user", "content": transcript}]
        else:
            messages = [{"role": "user", "content": input_data}]

        messages.insert(0, {
            "role": "system",
            "content": "You are a helpful assistant with enhanced parallel processing capabilities."
        })

        return chat_completions3(messages)

    def generate_fix(self, conversation_history, max_new_tokens=2048):
        processed_messages = []
        for msg in conversation_history:
            content = self._process_content(msg['content'][0])
            processed_messages.append({
                "role": msg['role'],
                "content": content
            })

        return chat_completions3(processed_messages)

    def _process_content(self, content_item):
        if content_item['type'] == 'audio':
            return self.asr(
                self._load_audio(content_item['audio']),
                generate_kwargs={
                    "language": "chinese",
                    "return_timestamps": True                }
            )['text'].strip()
        return content_item['text']