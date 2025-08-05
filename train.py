import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
import tiktoken
import matplotlib.pyplot as plt
import math
from model import BigramLanguageModel, batch_size, block_size, max_iters, eval_interval, learning_rate, device, eval_iters

ds = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)
ds = ds.take(1250000)  # 1.25M examples (~10% of openwebtext)
text = "\n".join([example["text"] for example in ds])
print(f"Loaded {len(text):,} characters")

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)

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
  model.eval() # switch to eval mode
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

# print number of parameters
print(f"Number of parameters: {sum(p.numel() for p in m.parameters()):,}")
    
# learning rate scheduler
warmup_iters = 2000
gradient_accumulation_steps = 8  # simulate batch_size=48 with gradient accumulation
lr_decay_iters = max_iters
min_lr = learning_rate / 10  # 6e-5

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate, weight_decay=0.1, betas=(0.9, 0.95))

# lists to store metrics for plotting
train_losses = []
val_losses = []
train_perplexities = []
val_perplexities = []
steps = []

best_val_loss = float('inf')
patience = 5  
no_improve_count = 0

for iter in range(max_iters):
    # update learning rate
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        train_loss = losses['train'] 
        val_loss = losses['val']
        train_ppl = torch.exp(train_loss).item()
        val_ppl = torch.exp(val_loss).item()
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        train_perplexities.append(train_ppl)
        val_perplexities.append(val_ppl)
        steps.append(iter)
        
        print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, train ppl {train_ppl:.2f}, val ppl {val_ppl:.2f}, lr {lr:.2e}")
        
        # early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            # save best model
            torch.save(m.state_dict(), "best_model.pt")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at step {iter} (best val loss: {best_val_loss:.4f})")
                break

    # gradient accumulation loop - avg gradient
    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(gradient_accumulation_steps):
        xb, yb = get_batch('train')
        logits, loss = m(xb, yb)
        loss = loss / gradient_accumulation_steps  # scale the loss
        loss.backward()
    
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0) # gradient clipping
    optimizer.step() # take a step to better weights

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# save the model
torch.save(m.state_dict(), "model_checkpoint.pt")
print("Model saved to model_checkpoint.pt")

# Create plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Cross-Entropy Loss
ax1.plot(steps, train_losses, label='Train Loss', color='blue')
ax1.plot(steps, val_losses, label='Val Loss', color='red')
ax1.set_xlabel('Steps')
ax1.set_ylabel('Cross-Entropy Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Perplexity (log scale)
ax2.semilogy(steps, train_perplexities, label='Train Perplexity', color='blue')
ax2.semilogy(steps, val_perplexities, label='Val Perplexity', color='red')
ax2.set_xlabel('Steps')
ax2.set_ylabel('Perplexity (log scale)')
ax2.set_title('Training and Validation Perplexity')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=150)
plt.show()