import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module): # causal self-attention layer.

    def __init__(self, config):
        super().__init__()
        assert config.D_MODEL % config.N_HEADS == 0
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.D_MODEL, 3 * config.D_MODEL)
        # output projection
        self.c_proj = nn.Linear(config.D_MODEL, config.D_MODEL)
        
        self.n_head = config.N_HEADS
        self.n_embd = config.D_MODEL
        self.dropout = config.DROPOUT
        
        # Causal mask: A buffer of ones, where the upper triangle is set to zero (masked).
        # I register it as a buffer so it's not a trainable parameter
        self.register_buffer("mask", torch.tril(torch.ones(config.BLOCK_SIZE, config.BLOCK_SIZE))
                                     .view(1, 1, config.BLOCK_SIZE, config.BLOCK_SIZE))
        
    def forward(self, x):
        B, T, C = x.size() # Batch size, sequence length (T), embedding dimension (C=D_MODEL)

        # first calculate Q, K, V for all heads, in a single forward pass
        qkv = self.c_attn(x) # qkv shape: (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embd, dim=2) # Splits into (B, T, C), (B, T, C), (B, T, C)
        
        # second, reshape and transpose for multi-head attention
        # k, q, v shape: (B, N_HEADS, T, D_HEAD) where D_HEAD = C / N_HEADS
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # third, calculate attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5)) # (B, N_HEADS, T, D_HEAD) @ (B, N_HEADS, D_HEAD, T) -> (B, N_HEADS, T, T)

         # fourth, apply causal masking
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        
        # fifth, calculate attention weights and apply dropout
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)

        # then apply attention weights to values
        y = att @ v # (B, N_HEADS, T, T) @ (B, N_HEADS, T, D_HEAD) -> (B, N_HEADS, T, D_HEAD)
        
        # finally re-assemble heads and project (c_proj)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Transpose back: (B, T, N_HEADS, D_HEAD)

        # Output projection
        y = self.c_proj(y)
        y = F.dropout(y, p=self.dropout, training=self.training)
        
        return y


class Block(nn.Module): # transformer decoder block (LayerNorm -> Causal Self-Attention -> LayerNorm -> MLP)

    def __init__(self, config):
        super().__init__()
        # LayerNorm 1 and Causal Self-Attention
        self.ln_1 = nn.LayerNorm(config.D_MODEL)
        self.attn = CausalSelfAttention(config)
        
        # LayerNorm 2 and MLP (Feed-Forward Network)
        self.ln_2 = nn.LayerNorm(config.D_MODEL)
        self.mlp = nn.Sequential(
            nn.Linear(config.D_MODEL, config.D_MODEL * config.FF_MULTIPLE),
            nn.GELU(), # GELU activation is common in modern transformers
            nn.Linear(config.D_MODEL * config.FF_MULTIPLE, config.D_MODEL),
            nn.Dropout(config.DROPOUT)
        )

    def forward(self, x):
        # causal self-attention with residual connection
        x = x + self.attn(self.ln_1(x))
        
        # mlp with residual connection
        x = x + self.mlp(self.ln_2(x))
        
        return x


class GPTModel(nn.Module): # decoder-only Transformer model
    
    def __init__(self, config, vocab_size, block_size):
        super().__init__()
        self.block_size = block_size
        
        # token embeddings (WTE) and positional embeddings (WPE)
        self.wte = nn.Embedding(vocab_size, config.D_MODEL)
        self.wpe = nn.Embedding(block_size, config.D_MODEL)
        self.drop = nn.Dropout(config.DROPOUT)
        
        # stack of L transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.N_LAYERS)])
        
        # final layernorm and the language model head (lm_head)
        self.ln_f = nn.LayerNorm(config.D_MODEL)
        self.lm_head = nn.Linear(config.D_MODEL, vocab_size, bias=False)
        
        print(f"Number of parameters: {sum(p.numel() for p in self.parameters())}")

    def forward(self, idx, targets=None):
        B, T = idx.size() # Batch size, Sequence length
        assert T <= self.block_size, f"Oh ooh, sequence length {T} exceeds block size {self.block_size}"
        
        # token and position embeddings
        tok_emb = self.wte(idx) # (B, T, D_MODEL)
        
        # 0, 1, 2, ..., T-1
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T)
        pos_emb = self.wpe(pos) # (T, D_MODEL)
        
        # combine token and dropout
        x = self.drop(tok_emb + pos_emb)
        
        # transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # final layernorm and lm head
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, V), V here being vocab_size
        
        loss = None
        if targets is not None:
            # reshape for cross-entropy loss:
            # logits: (B*T, V)
            # targets: (B*T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):# Takes a (B, T) tensor of indices and generates new tokens by autoregressive sampling.

        for _ in range(max_new_tokens):
            # wpe only holds embeddings up to block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond) # forward pass to get logits
            logits = logits[:, -1, :] # (B, V), focus only on the prediction for the very last token
            logits = logits / temperature # apply temperature (just 1 in my case)
            probs = F.softmax(logits, dim=-1) # apply softmax to get probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1), sample from the distribution
            idx = torch.cat((idx, idx_next), dim=1) # append sampled token to the sequence
            
        return idx