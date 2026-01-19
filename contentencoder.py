import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    ClapModel,
    ClapProcessor,
)
import math
import os

class HuBERTEncoder(nn.Module):
    def __init__(self, config, num_layers=24): #24 for hubert-large
        super().__init__()
        self.load_encoders_on_gpus=config.load_encoders_on_gpus
        if not config.use_finetuned_speech_encoder:
            self.model_path = config.speech_encoder_path  #ft. hubert
        else:
            self.model_path = config.speech_encoder_path_finetuned
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.speech_encoder = AutoModel.from_pretrained(self.model_path)
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    @torch.no_grad()
    def extract_features(self, batch, sampling_rate=16000):
        # audios: list of 1D tensors in range [-1,1]; returns (batch_size, seq_len, dim)
        audios = []
        for item in batch["audio"]:
            if item.is_cuda:
                item = item.cpu()
            audios.append(item.squeeze(0).detach().numpy())
        speech_lengths = [len(audio) for audio in audios]
        conv_kernels = [10, 3, 3, 3, 3, 2, 2]
        conv_strides = [5, 2, 2, 2, 2, 2, 2]
        for i in range(len(speech_lengths)):
            for k, s in zip(conv_kernels, conv_strides):
                speech_lengths[i] = math.floor((speech_lengths[i] - k) / s) + 1
        inputs = self.processor(audios, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        if not self.load_encoders_on_gpus:
            self.speech_encoder = self.speech_encoder.to("cpu")
        device = next(self.speech_encoder.parameters()).device
        input_values = inputs["input_values"].to(torch.bfloat16).to(device)
        attention_mask = inputs["attention_mask"].to(device)
        self.speech_encoder = self.speech_encoder.to(torch.bfloat16)
        outputs = self.speech_encoder(input_values = input_values, attention_mask = attention_mask)
        all_feats = outputs.last_hidden_state # (batch_size, seq_len, dim)
        feats = self.weights(all_feats)
        return feats, speech_lengths  # (batch_size, seq_len, dim)