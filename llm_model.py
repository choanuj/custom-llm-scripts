import torch
import torch.nn as nn
import math

# User-configurable variables
def get_user_config():
    print("Configure your LLM model:")
    vocab_size = int(input("Enter vocabulary size (e.g., 10000 for small models): ") or 10000)
    n_embd = int(input("Enter embedding dimension (e.g., 256): ") or 256)
    n_head = int(input("Enter number of attention heads (e.g., 8): ") or 8)
    n_layer = int(input("Enter number of transformer layers (e.g., 6): ") or 6)
    block_size = int(input("Enter max sequence length (e.g., 128): ") or 128)
    dropout = float(input("Enter dropout rate (e.g., 0.1): ") or 0.1)
    return {
        'vocab_size': vocab_size,
        'n_embd': n_embd,
        'n_head': n_head,
        'n_layer': n_layer,
        'block_size': block_size,
        'dropout': dropout
    }

# Multi-Head Self-Attention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_size = n_embd // n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        return self.dropout(self.proj(y))

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.attn = MultiHeadSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# GPT-like Language Model
class GPTLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config['vocab_size'], config['n_embd'])
        self.position_embedding = nn.Embedding(config['block_size'], config['n_embd'])
        self.blocks = nn.ModuleList([
            TransformerBlock(config['n_embd'], config['n_head'], config['block_size'], config['dropout'])
            for _ in range(config['n_layer'])
        ])
        self.ln_f = nn.LayerNorm(config['n_embd'])
        self.head = nn.Linear(config['n_embd'], config['vocab_size'])
        self.block_size = config['block_size']
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size, "Sequence length exceeds block size"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Note: For image generation, consider using pre-trained Stable Diffusion from Hugging Face.
# For video generation, explore VideoGAN or diffusion-based video models, which require
# separate architectures and significant computational resources.

if __name__ == "__main__":
    config = get_user_config()
    model = GPTLanguageModel(config)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    # Example: Save model configuration
    torch.save(config, 'model_config.pth')