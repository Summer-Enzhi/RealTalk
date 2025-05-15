from .base import VoiceAssistant
from transformers import AutoModel, AutoTokenizer
from transformers import WhisperFeatureExtractor
from loguru import logger
import torch
from .src_glm.speech_tokenizer.utils import extract_speech_token
from .src_glm.speech_tokenizer.modeling_whisper import WhisperVQEncoder
import soundfile as sf
import numpy as np
from resampy import resample

class GLMAssistant(VoiceAssistant):
    def __init__(self):
        model_path = 'THUDM/glm-4-voice-9b'
        self.glm_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=None,
            device_map="auto",  # 保持自动设备分配
            cache_dir='./cache',
            torch_dtype=torch.bfloat16,
        ).eval()
        self.glm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir='./cache')
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("THUDM/glm-4-voice-tokenizer", cache_dir='./cache')
        self.whisper_model = WhisperVQEncoder.from_pretrained("THUDM/glm-4-voice-tokenizer", cache_dir='./cache').eval().to("cuda")

    def _prepare_inputs(self, inputs):
        return {
            k: v.to(self.glm_model.device) if isinstance(v, torch.Tensor) else v 
            for k, v in inputs.items()
        }

    def generate_audio(
        self,
        audio,
        max_new_tokens=4096,
    ):
        audio_tokens = extract_speech_token(
            self.whisper_model, self.feature_extractor, [tuple([torch.from_numpy(audio['array']).unsqueeze(0), audio['sampling_rate']])]
        )[0]
        assert len(audio_tokens) != 0
        audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
        audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
        user_input = audio_tokens
        system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "

        inputs = f"<|system|>\n{system_prompt}"
        inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
        inputs = self.glm_tokenizer([inputs], return_tensors="pt")
        inputs = self._prepare_inputs(inputs)

        rtn = self.glm_model.generate(**inputs, max_new_tokens=max_new_tokens)[:, inputs.input_ids.size(1):]
        text_tokens = []
        audio_offset = self.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
        for item in rtn[0]:
            if item < audio_offset:
                text_tokens.append(item)
        logger.info(text_tokens)
        return self.glm_tokenizer.decode(text_tokens, ignore_special_tokens=False)

    def generate_text(
        self,
        text,
    ):
        history = []
        history.append({"role": "user", "content": text})
        user_input = text
        system_prompt = "User will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."
        inputs = f"<|system|>\n{system_prompt}"
        inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
        inputs = self.glm_tokenizer([inputs], return_tensors="pt")
        inputs = self._prepare_inputs(inputs)  # 替换原有设备分配方式

        rtn = self.glm_model.generate(**inputs, max_new_tokens=4096, synced_gpus=True)[:, inputs.input_ids.size(1):]
        text_tokens = []
        audio_offset = self.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
        for item in rtn[0]:
            if item < audio_offset:
                text_tokens.append(item)
        return self.glm_tokenizer.decode(text_tokens, ignore_special_tokens=False)
    
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
        audio_segments = []

        for msg in conversation_history:
            if msg['role'] == 'system':
                    audio_segments.append(
                        f"<|system|>\n{msg['content'][0]['text']}"
                    )
                    
            if msg['role'] == 'user' or msg['role'] == 'assistant' :
                for content in msg['content']:
                    if content['type'] == 'audio':
                        audio = self._load_audio(content['audio'])
                        audio_tokens = extract_speech_token(
                            self.whisper_model,
                            self.feature_extractor,
                            [(torch.from_numpy(audio).unsqueeze(0), 16000)]
                        )[0]
                        audio_segments.append(
                            f"<|{msg['role']}|>\n<|begin_of_audio|>" + 
                            "".join([f"<|audio_{x}|>" for x in audio_tokens]) +
                            "<|end_of_audio|>"
                        )
                    elif content['type'] == 'text':
                        audio_segments.append(f"<|{msg['role']}|>\n{content['text']}\n")

        inputs = "\n".join(audio_segments) + "\n"
        inputs += "<|assistant|>intention_analysis\n"

        processed_inputs = self.glm_tokenizer(
            [inputs], 
            return_tensors="pt"
        )
        processed_inputs = self._prepare_inputs(processed_inputs)  # 应用新方法

        generate_ids = self.glm_model.generate(
            **processed_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )[:, processed_inputs["input_ids"].size(1):]

        text_tokens = []
        audio_offset = self.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
        for token_id in generate_ids[0]:
            if token_id < audio_offset:
                text_tokens.append(token_id)
        
        return self.glm_tokenizer.decode(
            text_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )