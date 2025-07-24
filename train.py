import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
import tiktoken
import matplotlib.pyplot as plt
from model import BigramLanguageModel, batch_size, block_size, max_iters, eval_interval, learning_rate, device, eval_iters

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
text = "\n".join(ds["text"])

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
      losses[k] = loss.item()  # This is already per-token CE loss
    out[split] = losses.mean()  # Average of per-token losses
  model.train()
  return out

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# Print number of parameters
print(f"Number of parameters: {sum(p.numel() for p in m.parameters()):,}")
    
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# Lists to store metrics for plotting
train_losses = []
val_losses = []
train_perplexities = []
val_perplexities = []
steps = []

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        train_loss = losses['train'] 
        val_loss = losses['val']
        train_ppl = torch.exp(train_loss).item()
        val_ppl = torch.exp(val_loss).item()
        
        # Store metrics
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        train_perplexities.append(train_ppl)
        val_perplexities.append(val_ppl)
        steps.append(iter)
        
        print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, train ppl {train_ppl:.2f}, val ppl {val_ppl:.2f}")

    xb, yb = get_batch('train')

    # eval the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True) # zero gradients from prev step
    loss.backward() # find gradients
    optimizer.step() # take a step to better weights

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# Save the model
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