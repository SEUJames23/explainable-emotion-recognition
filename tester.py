from dataclasses import dataclass
from typing import Any, Tuple
import os, math, json, random
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers import AutoTokenizer
import json
from types import SimpleNamespace
from dataloader import AudioTextDataset, collate_fn
from speechencoder import HuBERTEncoder, CLAPAudioEncoder
from contentencoder import WhisperEncoder
from speechprojector import SpeechProjector
from contentprojector import ContentProjector
from speechllm import SpeechLLM
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from peft import PeftModel
import argparse
import pandas as pd
from tqdm import tqdm

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return SimpleNamespace(**cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--save_folder", type=str, default="prediction_results/")
    parser.add_argument("--save_file", type=str, default="pred.csv")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--eos_token_id", type=int, default=128009)
    parser.add_argument("--use_bfloat16", type=bool, default=True)
    
    args = parser.parse_args()
    config = load_config(args.config_path)
    traindataset = AudioTextDataset(config)
    train_dataloader = DataLoader(
        traindataset,
        batch_size=args.batch_size, 
        shuffle=False,   
        num_workers=4,           
        pin_memory=True,         
        collate_fn=collate_fn,   
        drop_last=False          
    )

    load_path = args.load_path
    model = SpeechLLM(config, args.device, load_path)
    model = model.to(torch.bfloat16)

    outputs_list = []
    tokenizer = AutoTokenizer.from_pretrained(config.llm_path) 

    with torch.inference_mode():
        for batch in tqdm(train_dataloader, desc="Generating predictions", ncols=100):
            out = model.generate(batch, args.max_new_tokens, args.temperature, args.top_k, args.eos_token_id, args.use_bfloat16)
            output_ids = out
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            print("outputs:", outputs)
            outputs_list.extend(outputs)

    if args.save_folder:
        os.makedirs(args.save_folder, exist_ok=True)
        import csv
        dfnames = pd.read_json(config.data_path, lines=True)["audio"]
        ids = [id for id in dfnames]

        with open(os.path.join(args.save_folder, args.save_file), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'asr'])
            for item1, item2 in zip(ids, outputs_list):
                writer.writerow([item1, item2])
