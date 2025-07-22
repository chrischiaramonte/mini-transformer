# Mini Transformer Language Model

a from scratch implementation of a Transformer-based language model trained on the WikiText-2 dataset using PyTorch.   
Implements byte-level BPE tokenization ('gpt-2' via 'tiktoken'), a 4-layer/6-head transformer, and next-token predicton

## Repository Structure
mini-transformer/  
├── model.py        # Hyperparameters + Transformer model definition  
├── train.py        # Tokenization, dataset loading, training loop, generation  
└── README.md  

## Hyperparameters
| Name           | Value | Meaning                                   |
| -------------- | ----- | ----------------------------------------- |
| batch\_size    | 32    | Sequences in parallel per optimization step|
| block\_size    | 128   | Context window (max sequence length)      |
| max\_iters     | 5000  | Training iterations                       |
| eval\_interval | 500   | Print losses every N steps                |
| learning\_rate | 3e-4  | AdamW learning rate                       |
| n\_embd        | 384   | Embedding / hidden vector size            |
| n\_head        | 6     | Attention heads per layer                 |
| n\_layer       | 4     | Number of blocks                          |
| dropout        | 0.2   | Dropout probability                       |

## Model Architecture
Embeddings:  
token (nn.Embedding(vocab_size, 384))  
position (nn.Embedding(block_size, 384))

Transformer Block (repeated n_layer times):  
x = x + SelfAttention(LayerNorm(x))  
x = x + FeedForward(LayerNorm(x))

Multi-head self-attention:  
6 heads, masked using lower triangular matrix

Feed-forward network:  
384 -> 1536 -> 384 with ReLU and dropout

LayerNorm before each sub-layer (self attention + feed forward)

Output head:  
Final layerNorm then Linear(384 -> vocab_size) -> logits (raw next-token scores)

## Data Pipeline
get_batch(split) randomly samples start positons and returns:  
- x: (batch_size, block_size) token encodings
- y: same tokens shifted by one (next token targets)

## Training
train.py loop:
1. Periodic evaluaton with estimate_loss() (averages over multiple batches)
2. Forward pass -> cross-entropy loss
3. Backward pass
4. Update parameters wiith AdamW

At the end it generates N tokens from an initial zero token

## Text Generation (model.generate)
1. Take the last block_size tokens of the current sequence as context
2. Forward pass -> logits (B, T, vocab_size)
3. Use latest token logits[:, -1, :]
4. Softmax probabilities
5. Sample one token per batch (torch.multinomial)
6. Append and repeat N times

## References 
- [Karpathy’s nanoGPT](https://github.com/karpathy/nanoGPT)  
- [WikiText-2 dataset](https://huggingface.co/datasets/wikitext)  
- [`tiktoken` library](https://github.com/openai/tiktoken)  
- **Attention Is All You Need** (Vaswani et al., 2017): the original Transformer paper introducing self‑attention. 

## Author
**Chris Chiaramonte**  
 