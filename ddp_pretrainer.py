import os
import math
import json
import argparse
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, distributed

from transformers import get_linear_schedule_with_warmup

from dataloader import AudioTextDataset, collate_fn
from speechllmflash import SpeechLLMFlash
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

import wandb
wandb.login(key="ec819c08fd85593b13fd70ac2195629a4e672c49")

def setup_ddp():
    """初始化 DDP 环境"""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")
    return local_rank, device


def cleanup_ddp():
    dist.destroy_process_group()


def build_lora_config(r=16, alpha=32, dropout=0.05, target='llm'):
    common_targets = ['q_proj','k_proj','v_proj','o_proj','up_proj','down_proj','gate_proj']
    if target == 'projector':
        target_modules = ['net.3', 'net.0', 'net.1', 'net.4']
    elif target == 'both':
        target_modules = common_targets + ['net.3', 'net.0', 'net.1', 'net.4']
    else:
        target_modules = common_targets

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias='none',
        target_modules=target_modules,
    )

def apply_lora(model, lora_cfg, target='llm'):
    if target in ('llm', 'both'):
        model.llm = get_peft_model(model.llm, lora_cfg)
    if target in ('projector', 'both'):
        for p in model.projector.parameters():
            p.requires_grad = True
    return model

def save_model(ddp_model, save_folder, step):
    save_dir = os.path.join(save_folder,"step_"+str(step))
    os.makedirs(save_dir, exist_ok=True)
    unwrapped = ddp_model.module  # DDP unwrap

    # 保存 LLM lora 权重
    if hasattr(unwrapped.llm, "save_pretrained"):
        unwrapped.llm.save_pretrained(save_dir)

    # 保存 projector
    projector_state = {}
    if hasattr(unwrapped, "speech_projector") and unwrapped.speech_projector is not None:
        projector_state["speech_projector"] = unwrapped.speech_projector.state_dict()
    if hasattr(unwrapped, "content_projector") and unwrapped.content_projector is not None:
        projector_state["content_projector"] = unwrapped.content_projector.state_dict()

    torch.save(projector_state, os.path.join(save_dir, "projector.pt"))
    print(f"✅ Model & projector saved to {save_dir}")

def train(config, args):
    local_rank, device = setup_ddp()
    print(f"[Rank {local_rank}] Device: {device}")

    # 初始化模型并移动到 GPU
    if args.train_from_ckpt=="True":
        load_path = args.load_path
        model = SpeechLLMFlash(config, device_id=device, load_path=load_path).to(device)
    else:
        model = SpeechLLMFlash(config, device_id=device).to(device)

    # LoRA
    if args.use_lora:
        lcfg = build_lora_config(r=args.lora_r, alpha=args.lora_alpha,
                                 dropout=args.lora_dropout, target=args.lora_target)
        apply_lora(model, lcfg, target=args.lora_target)

    # DDP
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    if local_rank == 0:
        print("DDP model ready!")

    # 数据集与分布式 sampler
    ds = AudioTextDataset(config)
    sampler = distributed.DistributedSampler(ds)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=8,  # 之前是4，CPU足够可以加大
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=4,  # 每个worker预取4个batch
    )

    # 优化器和 scheduler
    trainable_params = [p for p in ddp_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = math.ceil(len(dl) * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, args.warmup_steps, total_steps)

    ddp_model.train()

    # 计算总步数
    total_steps = args.epochs * len(dl)
    step = 0
    # 只有 local_rank 0 显示进度条
    if local_rank == 0:
        loop = tqdm(total=total_steps, desc="Training", unit="step")

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        
        scaler = torch.cuda.amp.GradScaler(enabled=args.bf16)
        for batch in dl:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.bf16):
                out = ddp_model(batch)
            scaler.scale(out.loss).backward()
            
            scaler.step(opt)
            scaler.update()
            sched.step()
            
            if local_rank == 0:
                # 更新进度条
                loop.update(1)
                loop.set_postfix(loss=f"{out.loss.item():.4f}")
                
                # 日志打印
                if step % args.logging_steps == 0 and step != 0:
                    print(f"[GPU {local_rank}] Epoch {epoch} Step {step} Loss {out.loss.item():.4f}")

                # 保存模型
                if step % args.save_steps == 0 and step != 0:
                    save_model(ddp_model, args.save_folder, step)

            step += 1

    if local_rank == 0:
        loop.close()

    cleanup_ddp()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return SimpleNamespace(**json.load(f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--bf16", action='store_true', default=True)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=2)
    parser.add_argument("--use_lora", type=bool, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target", type=str, default="llm")
    parser.add_argument("--train_from_ckpt", type=bool, default=False)
    parser.add_argument("--load_path", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config_path)
    train(config, args)
