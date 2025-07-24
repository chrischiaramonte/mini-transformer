import tiktoken

enc = tiktoken.get_encoding("gpt2")

# Check how special tokens are encoded
test_strings = ["<endOfText>", "<human>", "<bot>"]

for s in test_strings:
    tokens = enc.encode(s)
    print(f"{s} -> {tokens} ({len(tokens)} tokens)")
    for t in tokens:
        print(f"  Token {t}: '{enc.decode([t])}'")