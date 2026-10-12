from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

import time
import math

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
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
        self.NANOLLM_SCALE_INIT = 1 # For residual connections, scale the initialization of the weights by 1/sqrt(2 * n_layers)
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
        self.NANOLLM_SCALE_INIT = 1 # For residual connections, scale the initialization of the weights by 1/sqrt(2 * n_layers)

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
                std += (2 * self.model_args.n_layers) ** -0.5
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

        print(f"num decayed parameter tensors: {len(decay_params)}, num decayed parameters: {num_decay_params}")
        print(f"num non-decayed parameter tensors: {len(no_decay_params)}, num non-decayed parameters: {num_no_decay_params}")

        # Using Fused Adam optimizer from PyTorch
        use_fused = True if device.type == 'cuda' else False
        print(f"Using Fused Adam: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


# ---------------------------------------------------------------------------------------------------------------------------------------

import tiktoken 

# DataLoader
class DataLoaderLite:
    def __init__(self, batch_size, seq_len, device):
        self.B = batch_size
        self.T = seq_len
        self.device = device

        # At init load the entire dataset into memory
        with open("input.txt", "r") as f:
            text = f.read()
        
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded dataset with {len(self.tokens)} tokens")
        print(f"Number of batches: {len(self.tokens) // (self.B * self.T)}")

        # state
        self.current_position = 0

    def next_batch(self):
        if self.current_position + self.B * self.T + 1 >= len(self.tokens):
            # reset to start
            self.current_position = 0

        buf = self.tokens[self.current_position:self.current_position + self.B * self.T + 1]
        x = buf[:-1].view(self.B, self.T).to(self.device) # [B, T]
        y = buf[1:].view(self.B, self.T).to(self.device) # [B, T]

        self.current_position += self.B * self.T

        return x, y

# ---------------------------------------------------------------------------------------------------------------------------------------
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)

# Cosine Annealing Learning Rate Scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_iters = 10
max_steps = 50

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

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

train_loader = DataLoaderLite(batch_size=128, seq_len=1024, device=device)

# Using TensorFloat32 (TF32)
torch.set_float32_matmul_precision('high')

# Optimizer
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, device=device)

# Training loop
for iter in range(max_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()

    optimizer.zero_grad()

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits, loss = model(x, y)

    loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Determine and set the learning rate for this iteration
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000  # milliseconds

    tokens_per_second = (train_loader.B * train_loader.T) / (t1 - t0)

    if iter % 2 == 0:
        print(f"Step {iter}, Loss: {loss.item():.6f}, Norm: {norm.item():.4f}, Time: {dt:.2f} ms, Tokens/sec: {tokens_per_second:.2f}, LR: {lr:.6e}")


import sys; sys.exit(0)

# Sampling loop
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, my name is")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(5, 1).to(device) # [5, seq_len]

torch.manual_seed(42)
torch.cuda.manual_seed(42)

while tokens.size(1) < 100:
    with torch.no_grad():
        logits = model(tokens) # [batch_size, seq_len, vocab_size]
        logits = logits[:, -1, :] # [batch_size, vocab_size]
        probs = F.softmax(logits, dim=-1) # [batch_size, vocab_size]
        next_token = torch.multinomial(probs, num_samples=1) # [batch_size, 1]
        tokens = torch.cat((tokens, next_token), dim=1) # [batch_size, seq_len + 1]


# print the generated text
for i in range(5):
    generated_tokens = tokens[i, :100].tolist()
    text = enc.decode(generated_tokens)
    print(f"Generated text {i+1}: {text}")