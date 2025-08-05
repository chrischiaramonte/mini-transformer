# Mini Transformer Language Model

A from-scratch implementation of a Transformer-based language model trained on OpenWebText and fine-tuned on conversational data using PyTorch. Built as a learning project to understand transformer architecture, training dynamics, and chat model fine-tuning.

Implements byte-level BPE tokenization ('gpt-2' via 'tiktoken'), a 12-layer/12-head transformer (~125M params), and next-token prediction with chat capabilities.

## Repository Structure
mini-transformer/  
├── model.py              # Hyperparameters + Transformer model definition  
├── train.py              # Base model training on OpenWebText  
├── chat_train.py         # Fine-tuning on conversational data  
├── load_synthetic_persona.py     # Load chat data
├── chat.py               # Interactive chat interface  
└── README.md  

## Hyperparameters
| Name           | Value | Meaning                                   |
| -------------- | ----- | ----------------------------------------- |
| batch\_size    | 6     | Sequences in parallel per optimization step|
| block\_size    | 1024  | Context window (max sequence length)      |
| max\_iters     | 100000| Training iterations                       |
| eval\_interval | 500   | Print losses every N steps                |
| learning\_rate | 3e-4  | AdamW learning rate                       |
| n\_embd        | 768   | Embedding / hidden vector size            |
| n\_head        | 12    | Attention heads per layer                 |
| n\_layer       | 12    | Number of blocks                          |
| dropout        | 0.1   | Dropout probability                       |

## Model Architecture
Embeddings:  
token (nn.Embedding(vocab_size, 768))  
position (nn.Embedding(block_size, 768))

Transformer Block (repeated n_layer times):  
x = x + SelfAttention(LayerNorm(x))  
x = x + FeedForward(LayerNorm(x))

Multi-head self-attention:  
12 heads, masked using lower triangular matrix

Feed-forward network:  
768 -> 3072 -> 768 with ReLU and dropout

LayerNorm before each sub-layer (self attention + feed forward)

Output head:  
Final layerNorm then Linear(768 -> vocab_size) -> logits (raw next-token scores)

## Data Pipeline
get_batch(split) randomly samples start positions and returns:  
- x: (batch_size, block_size) token encodings
- y: same tokens shifted by one (next token targets)

## Training
**Base model (train.py):**
- Dataset: OpenWebText (1.25M examples via HuggingFace)
- Gradient accumulation: 8 steps (effective batch_size = 48)
- Learning rate schedule: linear warmup (2000 steps) + cosine decay
- Early stopping with patience = 5
- Gradient clipping at 1.0

**Chat fine-tuning (chat_train.py):**
- Dataset: Synthetic persona conversations with clean `<human>`, `<bot>`, `<endOfText>` format
- Improved data quality with consistent tokenization
- Loads best pre-trained model (`best_model.pt`)
- Optimized hyperparameters:
  - Lower learning rate: 2e-5 
  - Gradient accumulation: 16 steps (increased for training stability)
- **Achieved validation loss: 0.6953**

Training loop:
1. Periodic evaluation with estimate_loss() (averages over multiple batches)
2. Forward pass -> cross-entropy loss
3. Backward pass with gradient accumulation
4. Update parameters with AdamW (weight decay 0.1)

At the end it generates sample text to verify training

## Text Generation (model.generate)
1. Take the last block_size tokens of the current sequence as context
2. Forward pass -> logits (B, T, vocab_size)
3. Use latest token logits[:, -1, :]
4. Apply temperature scaling (controls randomness)
5. Optional top-k filtering (removes low-probability tokens)
6. Softmax probabilities
7. Sample one token per batch (torch.multinomial)
8. Append and repeat N times

**Generation Parameters:**
- Temperature: 0.8 (balanced coherence and creativity) 
- Top-k: 35 (nanoChatGPT inspired range of 35-50)
- Max tokens: 50 (prevents overly long responses)

## Chat Interface (chat.py)
- Interactive command-line interface using the fine-tuned model (`chat_model_best.pt`)
- Conversation format: `<human>{user_input}<endOfText><bot>{response}<endOfText>`
- Maintains conversation context throughout the session
- Type 'quit' to exit

**Usage:**
```bash
python chat.py
```

**Model Performance:**  
*Base Model (OpenWebText pretraining):*
- Initial loss: 10.97 → Final loss: ~3.87
- Training progression: 58,194 perplexity → ~48 perplexity 
- Trained for 12,500+ steps with cosine learning rate decay

*Chat Fine-tuned Model:*
- **Validation loss: 0.6953** 
- **Validation perplexity: 2.00**
- Dramatically improved conversational coherence

## Sample Conversation

```
User: Hello, how are you?
Bot: I'm doing well, thank you. How are you doing?

User: I'm doing good, it's nice to meet you!
Bot: You too,Thank you.

User: What do you like to do for fun?
Bot: I like to sing, watch TV, and eat cheese.
```

## Limitations

- **Context confusion**: Can lose track of topic over longer exchanges
- **Repetitive responses**: May fall into loops saying similar phrases
- **Empty responses**: Occasionally generates no output for certain inputs
- **Limited reasoning**: Lacks deep understanding beyond pattern matching
- **Model size constraints**: ~125M parameters limits knowledge and capabilities compared to larger models

## References 
- [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)  
- [nanoChatGPT](https://github.com/lamm-mit/nanoChatGPT) - 
- [OpenWebText dataset](https://huggingface.co/datasets/openwebtext)  
- [PersonaChat dataset](https://huggingface.co/datasets/persona_chat)
- [`tiktoken` library](https://github.com/openai/tiktoken)  
- **Attention Is All You Need** (Vaswani et al., 2017)

## Author
**Chris Chiaramonte**  
 