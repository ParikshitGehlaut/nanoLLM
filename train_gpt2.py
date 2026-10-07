from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

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

        # Causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(model_args.max_seq_len, model_args.max_seq_len).view(1, 1, model_args.max_seq_len, model_args.max_seq_len))) # [1, 1, seq_len, seq_len]

    def forward(self, x):
        batch_size, seq_len, dim = x.size() # [batch_size, seq_len, dim]

        qkv = self.c_attn(x) # [batch_size, seq_len, dim] --> [batch_size, seq_len, 3 * dim]
        q, k, v = qkv.split(self.dim, dim=2) # [batch_size, seq_len, dim] * 3

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2) # [batch_size, n_heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2) # [batch_size, n_heads, seq_len, head_dim]
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2) # [batch_size, n_heads, seq_len, head_dim]

        # Self Attention (materializes the large (seq_len, seq_len) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5) # [batch_size, n_heads, seq_len, seq_len]
        att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float('-inf')) # [batch_size, n_heads, seq_len, seq_len]
        att = F.softmax(att, dim=-1) # [batch_size, n_heads, seq_len, seq_len]
        
        y = att @ v # [batch_size, n_heads, seq_len, seq_len] @  [batch_size, n_heads, seq_len, head_dim] --> [batch_size, n_heads, seq_len, head_dim]

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
model = GPT(GPTConfig())
model.to(device)

train_loader = DataLoaderLite(batch_size=4, seq_len=32, device=device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training loop
for iter in range(1000):
    x, y = train_loader.next_batch()

    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print(f"Step {iter}, Loss: {loss.item()}")


import sys; sys.exit(0)

# Sampling loop
import tiktoken
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