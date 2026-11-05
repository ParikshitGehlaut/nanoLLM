from dataclasses import dataclass
import numpy as np

# PyTorch libraries
import torch
from torch import nn
from torch.nn import functional as F

import time
import math

# Distributed training libraries
import os
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "Distributed training requires CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ.get('RANK', -1))
    ddp_local_rank = int(os.environ.get('LOCAL_RANK', -1))
    ddp_world_size = int(os.environ.get('WORLD_SIZE', -1))
    device = torch.device(f'cuda:{ddp_local_rank}')
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    if master_process:
        print(f"Running DDP on AMD ROCm (HIP {torch.version.hip}) across {ddp_world_size} GPUs: {torch.cuda.get_device_name(0)}") 
else:
    # Non DDP run, on a single GPU or CPU
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if master_process:
            if torch.version.hip is not None:
                print(f"Running on AMD ROCm (HIP {torch.version.hip}): {torch.cuda.get_device_name(0)}")
            else:
                print(f"Running on NVIDIA CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

@dataclass
class GPTConfig:
    max_seq_len : int = 1024 # block size
    vocab_size : int = 50257
    n_layers : int = 12 # number of dense layers
    n_heads : int = 12 # number of attention heads
    dim : int = 768 # dim or n_embd, embedding dimension

# MHSA = Multi-Head Self Attention
class CausalSelfAttention(nn.Module):
    def __init__(self, model_args : GPTConfig):
        super().__init__()
        assert model_args.dim % model_args.n_heads == 0
        self.model_args = model_args
        # Key, Query, Value projections for all heads
        self.c_attn = nn.Linear(model_args.dim, 3 * model_args.dim)
        # Output projection
        self.c_proj = nn.Linear(model_args.dim, model_args.dim)
        # Regularization
        self.n_heads = model_args.n_heads
        self.c_proj.NANOLLM_SCALE_INIT = 1 # For residual connections, scale the initialization of the weights by 1/sqrt(2 * n_layers)
        self.dim = model_args.dim
        self.head_dim = self.dim // self.n_heads

    def forward(self, x):
        batch_size, seq_len, dim = x.size() # [batch_size, seq_len, dim]

        qkv = self.c_attn(x) # [batch_size, seq_len, dim] --> [batch_size, seq_len, 3 * dim]
        q, k, v = qkv.split(self.dim, dim=2) # [batch_size, seq_len, dim] * 3

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2) # [batch_size, n_heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2) # [batch_size, n_heads, seq_len, head_dim]
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2) # [batch_size, n_heads, seq_len, head_dim]

        # Using Flash Attention via PyTorch SDPA
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # [batch_size, n_heads, seq_len, head_dim]

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim) # [batch_size, seq_len, dim]
        y = self.c_proj(y) # [batch_size, seq_len, dim]

        return y

# MLP
class MLP(nn.Module):
    def __init__(self, model_args : GPTConfig):
        super().__init__()
        self.model_args = model_args
        self.c_fc = nn.Linear(model_args.dim, 4 * model_args.dim)
        self.c_proj = nn.Linear(4 * model_args.dim, model_args.dim)
        self.act = nn.GELU()
        self.c_proj.NANOLLM_SCALE_INIT = 1 # For residual connections, scale the initialization of the weights by 1/sqrt(2 * n_layers)

    def forward(self, x):
        x = self.c_fc(x) # [batch_size, seq_len, 4 * dim]
        x = self.act(x) # [batch_size, seq_len, 4 * dim]
        x = self.c_proj(x) # [batch_size, seq_len, dim]
        return x

# Decoder Layer
class Block(nn.Module):
    def __init__(self, model_args : GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(model_args.dim)
        self.attn = CausalSelfAttention(model_args)
        self.ln_2 = nn.LayerNorm(model_args.dim)
        self.mlp = MLP(model_args)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# Model 
class GPT(nn.Module):

    def __init__(self, model_args : GPTConfig):
        super().__init__()
        self.model_args = model_args

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(model_args.vocab_size, model_args.dim),
            wpe = nn.Embedding(model_args.max_seq_len, model_args.dim),
            h = nn.ModuleList([Block(model_args) for _ in range(model_args.n_layers)]),
            ln_f = nn.LayerNorm(model_args.dim),
        ))

        self.lm_head = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

        # Weight Sharing Scheme: The embedding weights and the output projection weights are shared.
        # Shape of wte: [vocab_size, dim], shape of lm_head.weight: [vocab_size, dim], In PyTorch, weight.shape = [out_features, in_features] 
        self.lm_head.weight = self.transformer.wte.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOLLM_SCALE_INIT'):
                std *= (2 * self.model_args.n_layers) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is shape : [batch_size, seq_len]
        batch_size, seq_len = idx.size()
        assert seq_len <= self.model_args.max_seq_len, f"Cannot forward sequence of length {seq_len}, block size is only {self.model_args.max_seq_len}"

        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device) # [seq_len]
        pos_emb = self.transformer.wpe(pos) # [seq_len, dim]
        tok_emb = self.transformer.wte(idx) # [batch_size, seq_len, dim]

        x = tok_emb + pos_emb # [batch_size, seq_len, dim]

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x) # [batch_size, seq_len, dim]
        logits = self.lm_head(x) # [batch_size, seq_len, vocab_size]
        loss = None 

        if targets is not None:
            # Logits are of shape [batch_size, seq_len, vocab_size], targets are of shape [batch_size, seq_len]
            # logits [batch_size, seq_len, vocab_size] --> [batch_size * seq_len, vocab_size]
            # targets [batch_size, seq_len] --> [batch_size * seq_len]
            # ignore_index=-1 means that the loss will ignore the positions where the target is -1 (padding)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # Start with all of the candidate parameters (that requires grad)
        param_dict = {name: param for name, param in self.named_parameters() if param.requires_grad}
        # Create optimizer groups for weight decay and no weight decay
        # Any parameters that is 2D (like matmuls + embeddings) will be weight decayed, all bias and LayerNorm weights will not be weight decayed
        decay_params = [param for name, param in param_dict.items() if param.ndim >= 2]
        no_decay_params = [param for name, param in param_dict.items() if param.ndim < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)

        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, num decayed parameters: {num_decay_params}")
            print(f"num non-decayed parameter tensors: {len(no_decay_params)}, num non-decayed parameters: {num_no_decay_params}")

        # Using Fused Adam optimizer from PyTorch
        use_fused = True if device.type == 'cuda' else False
        if master_process:
            print(f"Using Fused Adam: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


# ---------------------------------------------------------------------------------------------------------------------------------------

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

# DataLoader
class DataLoaderLite:
    def __init__(self, batch_size, seq_len, device, process_rank=0, world_size=1, split='train'):
        self.B = batch_size
        self.T = seq_len
        self.device = device
        self.process_rank = process_rank
        self.world_size = world_size
        assert split in {'train', 'val'}, "split must be either 'train' or 'val'"
        self.split = split

        # get the shard filenames
        data_root = 'climbmix_small/tokens/'
        shards = os.listdir(data_root)
        shards = [s for s in shards if self.split in s]
        shards.sort() # sort the shards to ensure consistent ordering
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards

        assert len(self.shards) > 0, f"No shards found for split {self.split} in {data_root}"
        if master_process:
            print(f"Found {len(self.shards)} shards for split {self.split}")

        # reset state for the DataLoader
        self.reset()

    def reset(self):
        self.current_shard_idx = 0
        self.tokens = load_tokens(self.shards[self.current_shard_idx]) # load the first shard
        self.current_position = self.B * self.T * self.process_rank # each process starts at a different position in the dataset

    def next_batch(self):
        buf = self.tokens[self.current_position:self.current_position + self.B * self.T + 1]
        x = buf[:-1].view(self.B, self.T).to(self.device) # [B, T]
        y = buf[1:].view(self.B, self.T).to(self.device) # [B, T]

        self.current_position += self.B * self.T * self.world_size # move to the next position for this process

        # if we have reached the end of the current shard, load the next shard
        if self.current_position + (self.B * self.T * self.world_size + 1) > len(self.tokens):
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shards) # wrap around to the first shard if we reach the end
            self.tokens = load_tokens(self.shards[self.current_shard_idx])
            self.current_position = self.B * self.T * self.process_rank

        return x, y

# ---------------------------------------------------------------------------------------------------------------------------------------
# Training loop

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Cosine Annealing Learning Rate Scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_iters = 50
max_steps = 1668 * 4 # 4 epochs of training on the dataset

def get_lr(it):
    # 1. Linear warmup for the first warmup_iters steps
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    # 2. If iter > lr_decay_iters, then return min_lr
    if it > max_steps:
        return min_lr
    # 3. Cosine annealing from max_lr to min_lr
    decay_ratio = (it - warmup_iters) / (max_steps - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges from 1 to 0
    return min_lr + coeff * (max_lr - min_lr)

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens 
B = 64 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, f"total_batch_size must be divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size) # each GPU will process B * T tokens, and there are ddp_world_size GPUs.

if master_process:
    print(f"Total desired batch size: {total_batch_size}")
    print(f"==> Calculated gradient accumulation steps: {grad_accum_steps} (micro batch size: {B}, sequence length: {T}, number of GPUs: {ddp_world_size})")


train_loader = DataLoaderLite(batch_size=B, seq_len=T, device=device, process_rank=ddp_rank, world_size=ddp_world_size, split='train')
val_loader = DataLoaderLite(batch_size=B, seq_len=T, device=device, process_rank=ddp_rank, world_size=ddp_world_size, split='val')

# Using TensorFloat32 (TF32)
torch.set_float32_matmul_precision('high')

# Create Model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

# Optimizer
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=min_lr, device=device)

# Create a log directory for saving checkpoints and logs
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'training_log.txt')
with open(log_file, 'w') as f:
    pass # just create an empty log file


for iter in range(max_steps):
    # Training step
    model.train()
    t0 = time.time()
    last_step = iter == max_steps - 1
    optimizer.zero_grad()
    loss_accum = 0.0
    # Doing gradient accumulation across grad_accum_steps micro batches
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # only sync gradients on the last micro step
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps # Normalize the loss because it is accumulated
        loss_accum += loss.detach()
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # average the loss across all GPUs

    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Determine and set the learning rate for this iteration
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000  # milliseconds

    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_second = tokens_processed / (t1 - t0)

    if master_process and (iter < 10 or iter % 10 == 0):
        with open(log_file, 'a') as f:
            f.write(f"Step {iter}| Loss: {loss_accum.item():.6f}| Norm: {norm.item():.4f}| Time: {dt:.2f} ms| Tokens/sec: {tokens_per_second:.2f}| LR: {lr:.6e}\n")

        if iter > 0 and iter % 5000 == 0 or last_step:
            # Save a checkpoint every 5000 iterations
            checkpoint_path = os.path.join(log_dir, f"checkpoint_iter_{iter}.pt")
            checkpoint = {
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iter': iter,
                'loss': loss_accum.item(),
            }
            torch.save(checkpoint, checkpoint_path)
            with open(log_file, 'a') as f:
                f.write(f"Saved checkpoint to {checkpoint_path}\n")

    # Validation step every 250 iterations
    if iter % 250 == 0 or last_step:
        model.eval()
        val_loader.reset() # reset the validation loader to start from the beginning

        with torch.no_grad():
            val_loss_accum = 0.0
            val_steps = 10
            for _ in range(val_steps):
                x, y = val_loader.next_batch()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits, val_loss = model(x, y)
                val_loss = val_loss / val_steps # Normalize the loss because it is accumulated
                val_loss_accum += val_loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG) # average the validation loss across all GPUs
        if master_process:
            with open(log_file, 'a') as f:
                f.write(f"Validation Loss: {val_loss_accum.item():.6f}\n")

    


if ddp:
    destroy_process_group()