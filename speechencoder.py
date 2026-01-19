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
    def __init__(self, config, num_layers=24): #24 for huerbt-large
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
        all_feats = outputs.hidden_state # (batch_size, seq_len, dim)
        feats = self.weights(all_feats)
        return feats, speech_lengths  # (batch_size, seq_len, dim)


class CLAPAudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.load_encoders_on_gpus = config.load_encoders_on_gpus
        self.clap_model_name = "/mmu-audio-ssd/frontend/youjun/pretrained_models/emoclap"
        self.model_path = config.speech_encoder_path_finetuned  # pre-trained emoclap
        self.clap_model_path = os.path.join(self.model_path, "model_weights.pt")
        self.processor = ClapProcessor.from_pretrained(self.clap_model_name)
        self.clapmodel = ClapModel.from_pretrained(self.clap_model_name)
        checkpoint = torch.load(self.clap_model_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        self.clapmodel.load_state_dict(state_dict, strict=False)
        self.speech_encoder = self.clapmodel.audio_model.audio_encoder
        # 冻结所有参数
        for param in self.speech_encoder.parameters():
            param.requires_grad = False

    def resample_waveform(self, waveform):
        waveform = waveform.detach().cpu()
        waveform = waveform.to(torch.float32)
        waveform = torchaudio.functional.resample(waveform, 16000, 48000)
        waveform = waveform.numpy().astype("float32")
        return waveform

    @torch.no_grad()
    def extract_features(self, batch, sampling_rate=48000):
        audios = [self.resample_waveform(item) for item in batch["audio"]]
        inputs = self.processor(audios=audios, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        if not self.load_encoders_on_gpus:
            self.speech_encoder = self.speech_encoder.to("cpu")
        device = next(self.speech_encoder.parameters()).device
        input_features = inputs["input_features"].to(torch.bfloat16).to(device)
        outputs = self.speech_encoder(input_features)
        feats = outputs.last_hidden_state # (B, 768, 2, 32)
        feats = torch.flatten(feats, 2).permute(0, 2, 1) # (B, 768, 2, 32) >> (B, 64, 768)
        speech_lengths = [64 for i in range(len(audios))]
        return feats, speech_lengths  # (B, 64, 768)
