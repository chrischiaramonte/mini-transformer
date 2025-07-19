from requests import head
import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from model import BigramLanguageModel, batch_size, block_size, max_iters, eval_interval, learning_rate, device, eval_iters

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

@torch.no_grad() # saves memory/compute by not building a backwards graph
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

model = BigramLanguageModel(vocab_size)
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