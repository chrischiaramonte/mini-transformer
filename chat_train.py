import torch
import torch.nn as nn
from torch.nn import functional as F
from model import BigramLanguageModel, batch_size, block_size, device
import matplotlib.pyplot as plt

data = torch.load("all_chat_tokens.pt")
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
        losses = torch.zeros(200)
        for k in range(200):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# load pretrained model
print("Loading pretrained model...")
import os
if not os.path.exists("model_checkpoint.pt"):
    print("ERROR: No pretrained model found!")
    print("Please run 'python train.py' first to create the base model.")
    exit(1)

vocab_size = 50257  # GPT-2 vocab size
model = BigramLanguageModel(vocab_size)
model.load_state_dict(torch.load("model_checkpoint.pt", map_location=device))
model = model.to(device)
print("Pretrained model loaded successfully!")

# fine-tune with lower learning rate
learning_rate = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

max_iters = 1000
eval_interval = 100
train_losses = []
val_losses = []
steps = []

print("Fine-tuning on chat data...")
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        train_loss = losses['train']
        val_loss = losses['val']
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        steps.append(iter)
        print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# save fine-tuned model
torch.save(model.state_dict(), "chat_model.pt")
print("Chat model saved")

plt.figure(figsize=(10, 5))
plt.plot(steps, train_losses, label='Train Loss', color='blue')
plt.plot(steps, val_losses, label='Val Loss', color='red')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Chat Fine-tuning - Train and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('chat_training.png')
plt.show()