from dataclasses import dataclass
from typing import List, Optional, Union, Dict
from speechencoder import HuBERTEncoder, CLAPAudioEncoder
from contentencoder import WhisperEncoder
from speechprojector import SpeechProjector
from contentprojector import ContentProjector
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    ClapModel,
    ClapProcessor,
)
from torch.cuda.amp import autocast, GradScaler
from peft import PeftModel
import os

class SpeechLLM(nn.Module):
    def __init__(self, config, device_id="cpu", load_path=None):
        super().__init__()
        self.device = device_id
        self.mode = config.mode
        self.llm_type = config.llm_type
        self.llm_path = config.llm_path
        self.speech_encoder_ds_rate = config.speech_encoder_ds_rate
        self.content_encoder_ds_rate = config.content_encoder_ds_rate
        self.freeze_speech_encoder = config.freeze_speech_encoder  #true or false
        self.freeze_content_encoder = config.freeze_content_encoder
        self.freeze_llm = config.freeze_llm
        self.id_speech_zhanweifu = config.id_speech_zhanweifu
        self.id_content_zhanweifu = config.id_content_zhanweifu
        self.id_assistant = config.id_assistant
        self.eot_id = config.eot_id

        # --- Speech encoder ---
        if config.speech_encoder_type.lower() == "hubert":
            self.speech_encoder = HuBERTEncoder(config)
        elif config.speech_encoder_type.lower() == "emoclap":
            self.speech_encoder = CLAPAudioEncoder(config)
        else:
            raise ValueError("speech_encoder_type must be 'hubert' or 'clap'")

        if self.freeze_speech_encoder:
            for p in self.speech_encoder.parameters():
                p.requires_grad = False

        # --- Content encoder ---
        self.content_encoder = WhisperEncoder(config)

        if self.freeze_content_encoder:
            for p in self.content_encoder.parameters():
                p.requires_grad = False

        # --- LLM decoder ---
        if self.llm_type.lower() == "llama":
            self.llm_path = self.llm_path
        elif self.llm_type.lower() == "qwen2.5":
            self.llm_path = self.llm_path
        else:
            raise ValueError("llm_type must be 'llama' or 'qwen2.5'")

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.llm_path,
            torch_dtype=torch.bfloat16,   # 改成 torch.float16 如果你是 A40
            device_map=None               # 若多卡，可改 "auto"
        ).to(self.device)

        if load_path!=None:
            print('-----load pre-trained ckpt to the model-----')
            self.lora_path = load_path
            self.llm = PeftModel.from_pretrained(self.llm, self.lora_path)
            self.llm = self.llm.merge_and_unload()

        if self.freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

        # --- Project to LLM text embedding ---
        self.speech_projector = SpeechProjector(config)
        self.content_projector = ContentProjector(config)

        if load_path!=None:
            ckpt = torch.load(os.path.join(load_path, "projector.pt"), map_location="cpu")
            if "speech_projector" in ckpt:
                self.speech_projector.load_state_dict(ckpt["speech_projector"])
            else:
                print("!!!There is no speech projector weights in "+load_path)
            if "content_projector" in ckpt:
                self.content_projector.load_state_dict(ckpt["content_projector"])
            else:
                print("!!!There is no content projector weights in "+load_path)

    @torch.no_grad()
    def _encode_speech(self, batch):
        feats, speech_lengths = self.speech_encoder.extract_features(batch)  # (B, T_s, D_in)
        feats = feats.to(self.device)
        self.speech_projector = self.speech_projector.to(self.device)
        speech_feats, speech_mu, speech_var = self.speech_projector(feats)  # (B, T_s, D_llm)
        new_speech_lengths = [speech_length // self.speech_encoder_ds_rate for speech_length in speech_lengths]
        feats = [feats[i, :speech_lengths[i]] for i in range(len(feats))]
        return speech_feats, speech_mu, speech_var

    @torch.no_grad()
    def _encode_content(self, batch):
        feats, content_lengths = self.content_encoder.extract_features(batch)
        feats = feats.to(self.device)
        self.content_projector = self.content_projector.to(self.device)
        content_feats, content_mu, content_var = self.content_projector(feats)
        new_content_lengths = [content_length // self.content_encoder_ds_rate for content_length in content_lengths]
        feats = [feats[i, :content_lengths[i]] for i in range(len(feats))]
        return content_feats, content_mu, content_var

    def prepare_inputs_labels_for_speech_and_text(self, batch):
        input_ids = batch['input_ids']  # B * (L_p)
        max_length = 0
        for i in range(len(input_ids)):
            if batch['audio'][i] != None:
                speech_feats_i = self._encode_speech({'audio': batch['audio'][i]})[0]
                content_feats_i = self._encode_content({'audio': batch['audio'][i]})[0]
            else:
                speech_feats_i = None
                content_feats_i = None
            length_speech_feats_i = len(speech_feats_i) if speech_feats_i != None else 0
            length_content_feats_i = len(content_feats_i) if content_feats_i != None else 0
            length_input_ids_i = len(input_ids[i])
            length_i = length_speech_feats_i+length_content_feats_i+length_input_ids_i-2 #two zhanweifu
            if length_i > max_length:
                max_length = length_i

        new_input_ids = torch.ones((len(input_ids), max_length), dtype=torch.long, device=self.device)*self.eot_id  ##pad using <|eot_id|>
        new_input_embeds = self.llm.get_decoder().embed_tokens(new_input_ids)
        new_labels = new_input_ids.clone()
        attention_mask = torch.zeros((len(input_ids), max_length), dtype=torch.long, device=self.device)
        position_ids = torch.zeros((len(input_ids), max_length), dtype=torch.long, device=self.device)

        info_loss = 0
        for i in range(len(input_ids)):
            if batch['audio'][i] != None:
                speech_feats_i, speech_mu_i, speech_var_i = self._encode_speech({'audio': batch['audio'][i]})[0]
                content_feats_i, content_mu_i, content_var_i = self._encode_content({'audio': batch['audio'][i]})[0]
                info_loss_i = -0.5 *(1 + torch.log(speech_var_i) - speech_mu_i.pow(2) - speech_var_i) -0.5 *(1 + torch.log(content_var_i) - content_mu_i.pow(2) - content_var_i)
            else:
                speech_feats_i = None
                content_feats_i = None
                info_loss_i = 0

            info_loss += info_loss_i
            input_ids_i = input_ids[i].to(self.device)
            length_speech_feats_i = len(speech_feats_i) if speech_feats_i != None else 0
            length_content_feats_i = len(content_feats_i) if content_feats_i != None else 0
            # print("length_speech_feats_i:", length_speech_feats_i)
            # print("length_content_feats_i:", length_content_feats_i)
            length_input_ids_i = len(input_ids_i)
            # print("length_input_ids_i:", length_input_ids_i)
            length_i = length_speech_feats_i+length_content_feats_i+length_input_ids_i-2 #two zhanweifu
            idx_speech_zhanweifu = (input_ids_i == self.id_speech_zhanweifu).nonzero(as_tuple=True)[0].item()
            idx_content_zhanweifu = (input_ids_i == self.id_content_zhanweifu).nonzero(as_tuple=True)[0].item()
            idx_assistant = (input_ids_i == self.id_assistant).nonzero(as_tuple=True)[0].item()
            length_prompt = idx_assistant+length_speech_feats_i+length_content_feats_i-1 #idx of an element in a tensor starts from 0
            new_labels[i, length_prompt:length_i] = input_ids_i[idx_assistant+1:length_input_ids_i]

            input_embeds_i = self.llm.get_decoder().embed_tokens(input_ids_i)
            left_i = input_embeds_i[:idx_speech_zhanweifu]
            middle_i = input_embeds_i[idx_speech_zhanweifu+1:idx_content_zhanweifu]
            right_i = input_embeds_i[idx_content_zhanweifu+1:]
            if speech_feats_i != None:
                input_embeds_i = torch.cat([left_i, speech_feats_i, middle_i, content_feats_i, right_i], dim=0)
            else:
                input_embeds_i = torch.cat([left_i, middle_i, right_i], dim=0)
            new_input_embeds[i, :length_i, :] = input_embeds_i
            new_labels[i, :length_prompt] = -100
            attention_mask[i, :length_i] = True
            position_ids[i, :length_i] = torch.arange(0, length_i)
        return {"position_ids": position_ids, "attention_mask": attention_mask, "inputs_embeds": new_input_embeds, "labels": new_labels, "info_loss": info_loss}

    def forward(self, batch):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = self.prepare_inputs_labels_for_speech_and_text(batch)
            position_ids = out['position_ids']
            attention_mask = out['attention_mask']
            inputs_embeds = out['inputs_embeds']
            labels = out['labels']
            info_loss = out["info_loss"]
            out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, position_ids=position_ids)
            out.loss += 0.001*info_loss
        return out

    @torch.no_grad()
    def generate_ori(self,batch):
        """Greedy/beam sampling with speech prefix + textual prompt.

        Provide `prompt_input_ids` already tokenized by your tokenizer.
        """
        self.eval()
        B = len(wavs)
        speech_tokens = self._encode_speech(wavs, sampling_rate)  # (B, T_s, D)
        prompt_embeds = self.llm.get_input_embeddings()(prompt_input_ids.to(self.device))
        inputs_embeds = torch.cat([speech_tokens, prompt_embeds], dim=1)

        attn_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=self.device)
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        batch,
        max_new_tokens: int = 1500,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_token_id: int = 128009,
        use_bfloat16: bool = True,
    ):
        device = self.device

        out = self.prepare_inputs_labels_for_speech_and_text(batch)
        inputs_embeds = out['inputs_embeds'].to(device)
        attention_mask = out['attention_mask'].to(device)
        position_ids = out['position_ids'].to(device)
        batch_size = inputs_embeds.size(0)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        generated_tokens = []

        # bfloat16 or fp32
        dtype_ctx = autocast if use_bfloat16 else torch.no_grad

        with dtype_ctx(dtype=torch.bfloat16 if use_bfloat16 else None):
            # first-step forward to obtain past_key_values
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values

            # fist token
            logits = outputs.logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            if temperature == 0 or top_k <= 1:
                next_token = logits.argmax(-1)
            else:
                # top-k sampling
                values, indices = torch.topk(logits, top_k)
                probs = torch.softmax(values, dim=-1)
                next_token = indices[torch.arange(batch_size, device=device), torch.multinomial(probs, 1).squeeze(-1)]
            generated_tokens.append(next_token.clone())

            # send to looping steps
            for step in range(1, max_new_tokens):
                # current step embedding
                next_embeds = self.llm.get_input_embeddings()(next_token).unsqueeze(1)

                # update attention mask & position_ids
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(batch_size, 1, device=device, dtype=torch.long)],
                    dim=1
                )
                position_ids = torch.arange(attention_mask.shape[1], device=device).unsqueeze(0).repeat(batch_size, 1)

                # predict next token
                outputs = self.llm(
                    inputs_embeds=next_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                if temperature == 0 or top_k <= 1:
                    next_token = logits.argmax(-1)
                else:
                    values, indices = torch.topk(logits, top_k)
                    probs = torch.softmax(values, dim=-1)
                    next_token = indices[torch.arange(batch_size, device=device), torch.multinomial(probs, 1).squeeze(-1)]

                generated_tokens.append(next_token.clone())

                # stop at eos
                finished |= next_token.eq(eos_token_id)
                if finished.all():
                    break

        # concat tokens
        generated_tokens = torch.stack(generated_tokens, dim=1)  # [bs, gen_len]
        return generated_tokens
