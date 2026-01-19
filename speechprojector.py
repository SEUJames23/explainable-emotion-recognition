import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeechProjector(nn.Module):
    def __init__(self, config, input_dim_ve=1024, latent_dim_ve=1024):
        super().__init__()
        # information bottleneck
        self.enc1 = torch.nn.Linear(input_dim_ve, input_dim_ve)
        self.enc2 = torch.nn.Linear(input_dim_ve, input_dim_ve)
        self.mu = torch.nn.Linear(input_dim_ve, latent_dim_ve)
        self.var = torch.nn.Linear(input_dim_ve, latent_dim_ve)
        # speech projector
        self.d_speech_encoder = config.d_speech_encoder
        self.d_llm = config.d_llm
        self.k = config.speech_encoder_ds_rate
        self.linear1 = nn.Linear(self.d_speech_encoder * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.d_llm)

    def forward(self, feats):
        # feats: torch.Size([batch_size, seq_len, dim]) >> torch.Size([batch_size, seq_len//k, d_llm])
        o = F.gelu(self.enc1(feats))
        o = F.gelu(self.enc2(o))
        mu = self.mu(o)
        var = F.softplus(self.var(o)) # to generate positive values
        std = var ** 0.5
        eps = torch.randn_like(std)
        feats = mu + std * eps
        batch_size, seq_len, dim = feats.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            feats = feats[:, :-num_frames_to_discard, :]
        seq_len = feats.size(1)

        feats = feats.contiguous()
        feats = feats.view(batch_size, seq_len // self.k, dim * self.k)
        feats = self.linear1(feats)
        feats = self.relu(feats)
        feats = self.linear2(feats)
        return feats, mu, var