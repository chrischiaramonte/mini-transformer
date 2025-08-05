import torch
import torch.nn as nn
from torch.nn import functional as F
from model import BigramLanguageModel, batch_size, block_size, device
import matplotlib.pyplot as plt
import math

data = torch.load("synthetic_persona_tokens.pt")
print(f"Loaded {len(data):,} tokens")

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(50)  # faster eval
        for k in range(50):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        avg_loss = losses.mean()
        perplexity = torch.exp(avg_loss)
        out[split] = avg_loss
        out[f'{split}_perplexity'] = perplexity
    model.train()
    return out

# load pretrained model
print("Loading pretrained model...")

vocab_size = 50257  # GPT-2 vocab size
model = BigramLanguageModel(vocab_size)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model = model.to(device)

print("Pretrained model loaded successfully!")

# fine-tune parameters - optimized for high-quality data
base_lr = 2e-5  # optimal learning rate for clean synthetic data
gradient_accumulation_steps = 16  # increased for training stability

optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01, betas=(0.9, 0.95))

max_iters = 5000  # increased iterations for better convergence
eval_interval = 100  # balanced evaluation frequency 
train_losses = []
val_losses = []
train_perplexities = []
val_perplexities = []
steps = []

best_val_loss = float('inf')

print("Fine-tuning on chat data...")
for iter in range(max_iters):
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        train_loss = losses['train']
        val_loss = losses['val']
        train_ppl = losses['train_perplexity']
        val_ppl = losses['val_perplexity']
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        train_perplexities.append(train_ppl.item())
        val_perplexities.append(val_ppl.item())
        steps.append(iter)
        print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, train ppl {train_ppl:.2f}, val ppl {val_ppl:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "chat_model_best.pt")

    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(gradient_accumulation_steps):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        loss = loss / gradient_accumulation_steps
        loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

# save fine-tuned model
torch.save(model.state_dict(), "chat_model.pt")
print(f"Chat model saved. Best val loss: {best_val_loss:.4f}")

plt.figure(figsize=(15, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(steps, train_losses, label='Train Loss', color='blue')
plt.plot(steps, val_losses, label='Val Loss', color='red')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Chat Fine-tuning - Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Perplexity plot
plt.subplot(1, 2, 2)
plt.plot(steps, train_perplexities, label='Train Perplexity', color='blue')
plt.plot(steps, val_perplexities, label='Val Perplexity', color='red')
plt.xlabel('Steps')
plt.ylabel('Perplexity')
plt.title('Chat Fine-tuning - Perplexity')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chat_training.png')
plt.show()