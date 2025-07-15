from requests import head
import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset

# hyper parameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
n_layer = 4
# -------------

torch.manual_seed(1337)

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
text = "\n".join(ds["text"])
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # for each character in a string
decode = lambda l: ''.join([itos[i] for i in l]) # for each integer in a list of integers

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% of the data
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # tensor of random ints
    x = torch.stack([data[i:i+block_size] for i in ix]) # 2d tensor each row is a batch
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # we wont call backpropogation so this is memory efficient
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

# single attention head
class Head(nn.Module):
    def __init__(self, head_size):
      super().__init__()
      self.key = nn.Linear(n_embd, head_size, bias=False)
      self.query = nn.Linear(n_embd, head_size, bias=False)
      self.value = nn.Linear(n_embd, head_size, bias=False)
      self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

      self.dropout = nn.Dropout(dropout)

    def forward(self, x):
       B, T, C = x.shape
       k = self.key(x) # (B, T, C)
       q = self.query(x) # (B, T, C)

      # compute attention score (how much i relates with j)
       wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, T) 
       wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
       wei = F.softmax(wei, dim=-1) # (B, T, T)
       wei = self.dropout(wei)

      # compute new vectors for tokens
       v = self.value(x) # (B, T, C)
       out = wei @ v # (B, T, C)
       return out
    

class MultiHeadAttenton(nn.Module):
    def __init__(self, n_head, head_size):
       super().__init__()
       self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
       self.proj = nn.Linear(n_embd, n_embd)
       self.dropout = nn.Dropout(dropout)

    def forward(self, x):
      out = torch.cat([h(x) for h in self.heads], dim=-1) # concat over C dim
      out = self.proj(out) # informaton was still partitioned
      return out
   

class FeedForward(nn.Module):
    def __init__(self, n_embd):
       super().__init__()
       self.net = nn.Sequential(
          nn.Linear(n_embd, 4 * n_embd), # expand size to making learning more rich
          nn.ReLU(),
          nn.Linear(4 * n_embd, n_embd),
          nn.Dropout(dropout)
       )

    def forward(self, x):
       return self.net(x)
    

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
       super().__init__()
       head_size = n_embd // n_head
       self.sa = MultiHeadAttenton(n_head, head_size)
       self.ffwd = FeedForward(n_embd)
       self.ln1 = nn.LayerNorm(n_embd)
       self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self):
      super().__init__()
      self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # random embedding table, each row is a learned vector
      self.position_embedding_table = nn.Embedding(block_size, n_embd) 
      self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
      self.ln_f = nn.LayerNorm(n_embd) # final
      self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
      B, T = idx.shape

      tok_embd = self.token_embedding_table(idx) # (B, T, C) C is each learned vector for that token
      pos_embd = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
      x = tok_embd + pos_embd
      x = self.blocks(x)

      logits = self.lm_head(x) # (B, T, vocab_size) raw score for each of the tokens possible next token

      if targets is None:
        loss = None
      else:
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        
      return logits, loss # have a logic vector of input with vocab_size

    def generate(self, idx, max_new_tokens): # for each batch continue generating tokens
      for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, loss = self(idx_cond) # calls foward
        logits = logits[:, -1, :] # becomes (B, C) only last T position -> predicting 9th token
        probs = F.softmax(logits, dim=-1) 
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) adds randomness to choosing next token for each batch
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
      return idx
    
model = BigramLanguageModel()
m = model.to(device)
    
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    # eval the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True) # zero gradients from prev step
    loss.backward() # find gradients
    optimizer.step() # take a step to better weights

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

