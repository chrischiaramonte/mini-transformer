import tiktoken

enc = tiktoken.get_encoding("gpt2")
print(f"GPT-2 vocab size: {enc.n_vocab}")