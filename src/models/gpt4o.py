from .base import VoiceAssistant
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import transformers
import torch
import io
import base64
from openai import OpenAI
import soundfile as sf
import librosa
import http.client
import json
import time

API_SECRET_KEY = "Your API Secret Key"
BASE_URL = "Your base url"

class GPT4oAssistant(VoiceAssistant):
    def __init__(self):
        self.client = OpenAI()
        self.model_name = "gpt-4o-audio-preview"

    def generate_audio(
        self,
        audio,
        max_new_tokens=2048,
    ):
        # Write the audio data to an in-memory buffer in WAV format
        buffer = io.BytesIO()
        sf.write(buffer, audio['array'], audio['sampling_rate'], format='WAV')
        buffer.seek(0)  # Reset buffer position to the beginning

        # Read buffer as bytes and encode in base64
        wav_data = buffer.read()
        encoded_string = base64.b64encode(wav_data).decode('utf-8')

        completion = self.client.chat.completions.create(
            model=self.model_name,
            modalities=["text"],
            max_tokens=max_new_tokens,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who tries to help answer the user's question."},
                {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": encoded_string, "format": 'wav'}}]},
            ]
        )

        return completion.choices[0].message.content


class GPT4oMiniAssistant(VoiceAssistant):
    def __init__(self):
        self.client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
        self.model_name = "gpt-4o-mini-audio-preview"
        
    def generate_fix(
        self,
        conversation_history,
        max_new_tokens=512,
    ):
        def to_base64(audio):
            # Convert audio array to bytes
            buffer = io.BytesIO()
            sf.write(buffer, audio[0], audio[1], format='WAV')
            buffer.seek(0)
            wav_data = buffer.read()
            # Encode bytes to base64
            encoded_string = base64.b64encode(wav_data).decode('utf-8')
            return encoded_string
        
        his = []
        for message in conversation_history:
            if message['content'][0]['type'] == 'audio':
                audio_file = message['content'][0]['audio']
                # raw_audio = self._load_audio(audio_file)
                # processed_audio = self._prepare_audio_input(raw_audio)
                processed_audio = librosa.load(audio_file, sr=16000, mono=True)
                encoded_string = to_base64(processed_audio)
                his.append({"role":message['role'],
                            "content":[{"type": "input_audio", "input_audio": {"data": encoded_string, "format": 'wav'}}]})
            else:
                his.append({"role":message['role'],
                            "content":message['content'][0]['text']})
        
        # completion = self.client.chat.completions.create(
        #     model=self.model_name,
        #     modalities=["text"],
        #     max_tokens=max_new_tokens,
        #     messages=his
        # )

        for _ in range(25):
            conn = None
            try:
                # 创建连接和准备请求
                conn = http.client.HTTPSConnection("api.chatfire.cn")
                payload = json.dumps({
                    "model": self.model_name,
                    "messages": his  # 假设his是实例变量
                }, ensure_ascii=False).encode('utf-8')
                
                headers = {
                    'Content-Type': 'application/json; charset=utf-8',
                    'Authorization': 'Bearer ' # your Authorization
                }
                
                # 发送请求
                conn.request("POST", "/v1/chat/completions", payload, headers)
                res = conn.getresponse()
                
                # 检查HTTP状态码
                if res.status != 200:
                    print(json.loads(res.read().decode("utf-8")))
                    raise Exception(f"API返回错误状态码：{res.status}")
                
                # 解析响应
                data = res.read().decode("utf-8")
                json_data = json.loads(data)
                
                # 验证响应结构
                if 'choices' not in json_data or not json_data['choices']:
                    raise Exception("响应中缺少choices字段")
                if 'message' not in json_data['choices'][0] or 'content' not in json_data['choices'][0]['message']:
                    raise Exception("响应消息结构异常")
                
                # 获取内容
                content = json_data['choices'][0]['message']['content']
                print(f'GPT-4o-mini：{content}')
                return content
                
            except Exception as e:
                print(f"请求失败（剩余尝试次数：{24-_}），错误信息：{str(e)}")
                time.sleep(5)
                
            finally:
                if conn:
                    conn.close()
        
        # 所有尝试失败后返回空字符串
        return ""
        