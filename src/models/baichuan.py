import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import ujson
from .base import VoiceAssistant
import tempfile
import soundfile as sf
import os
from huggingface_hub import snapshot_download


class BaichuanAssistant(VoiceAssistant):
    def __init__(self):
        self.sampling_rate = 24000
        self.role_prefix = {
            'system': '<B_SYS>',
            'user': '<C_Q>',
            'assistant': '<C_A>',
            'audiogen': '<audiotext_start_baichuan>'
        }
        self.load_model()
        self.model.training = False
        self.model.bind_processor(self.tokenizer, training=False, relative_path="/")
        self.audio_start_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audio_start_token_id)
        self.audio_end_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audio_end_token_id)
        self.special_token_partten = re.compile('<\|endoftext\|>|<audiogen_start_baichuan>|<audiogen_end_baichuan>')

    def load_model(self):
        raise NotImplementedError

    def preprocess_messages(self, messages):
        text = ""
        for i, msg in enumerate(messages):
            text += self.role_prefix[msg['role']]
            text += msg['content']
        text += self.role_prefix["assistant"]
        return text

    def generate_audio(
        self,
        audio,
        max_new_tokens=2048,
    ):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".wav") as temp_file:
            temp_filename = temp_file.name
            sf.write(temp_file.name, audio['array'], audio['sampling_rate'], format='wav')

        g_history = []

        g_history.append({
            "role": "system",
            "content": "You are a helpful assistant who tries to help answer the user's question."
        })

        g_history.append({
            "role": "user",
            "content": self.audio_start_token + ujson.dumps({'path': temp_filename}, ensure_ascii=False) + self.audio_end_token
        })
        message = self.preprocess_messages(g_history)
        pret = self.model.processor([message])
        plen = pret.input_ids.shape[1]
        ret = self.model.generate(
            pret.input_ids.cuda(),
            attention_mask=pret.attention_mask.cuda(),
            audios=pret.audios.cuda() if pret.audios is not None else None,
            encoder_length=pret.encoder_length.cuda() if pret.encoder_length is not None else None,
            bridge_length=pret.bridge_length.cuda() if pret.bridge_length is not None else None,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            stop_strings=['<|endoftext|>'],
            do_sample=True, temperature=0.8, top_k=20, top_p=0.85, repetition_penalty=1.1, return_dict_in_generate=True,
        )
        text_segment = self.tokenizer.decode(ret.sequences[0, plen:])
        full_text = re.sub(self.special_token_partten, '', text_segment)

        return full_text
    def _prepare_audio_input(self, audio_path):
        return self.audio_start_token + ujson.dumps({'path': audio_path}, ensure_ascii=False) + self.audio_end_token

    def preprocess_messages(self, conversation_history):
        text = ""
        for msg in conversation_history:
            text += self.role_prefix[msg['role']]
            
            for content in msg['content']:
                if content['type'] == 'audio':
                    audio_representation = self._prepare_audio_input(content['audio'])
                    text += audio_representation
                elif content['type'] == 'text':
                    text += content['text']
                
        text += self.role_prefix["assistant"]
        return text

    def generate_fix(self, conversation_history, max_new_tokens=2048):
        formatted_text = self.preprocess_messages(conversation_history)
        processed_input = self.model.processor([formatted_text])
        input_length = processed_input.input_ids.shape[1]
        
        generation_config = {
            'input_ids': processed_input.input_ids.cuda(),
            'attention_mask': processed_input.attention_mask.cuda(),
            'audios': processed_input.audios.cuda() if processed_input.audios is not None else None,
            'encoder_length': processed_input.encoder_length.cuda() if processed_input.encoder_length is not None else None,
            'bridge_length': processed_input.bridge_length.cuda() if processed_input.bridge_length is not None else None,
            'tokenizer': self.tokenizer,
            'max_new_tokens': max_new_tokens,
            'stop_strings': ['<|endoftext|>'],
            'do_sample': False,
            'temperature': 0,
            'return_dict_in_generate': True
        }
        
        generation_result = self.model.generate(**generation_config)
        
        generated_ids = generation_result.sequences[0, input_length:]
        decoded_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        clean_text = re.sub(self.special_token_partten, '', decoded_text).strip()
        
        return clean_text

class BaichuanOmniAssistant(BaichuanAssistant):
    def load_model(self):
        if not os.path.exists("./cache/Baichuan-Omni-1d5"):
            snapshot_download(
                repo_id="baichuan-inc/Baichuan-Omni-1d5",
                local_dir="./cache/Baichuan-Omni-1d5",
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            './cache/Baichuan-Omni-1d5', trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto'
        )
        self.tokenizer = AutoTokenizer.from_pretrained('./cache/Baichuan-Omni-1d5', trust_remote_code=True)


class BaichuanAudioAssistant(BaichuanAssistant):
    def load_model(self):
        if not os.path.exists("./cache/Baichuan-Audio-Instruct"):
            snapshot_download(
                repo_id="baichuan-inc/Baichuan-Audio-Instruct",
                local_dir="./cache/Baichuan-Audio-Instruct",
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            "./cache/Baichuan-Audio-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto'
        )
        self.tokenizer = AutoTokenizer.from_pretrained("./cache/Baichuan-Audio-Instruct", trust_remote_code=True)