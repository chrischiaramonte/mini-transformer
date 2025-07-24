import torch
import tiktoken
from pathlib import Path

def load_and_tokenize_chat_data(file_path):    
    
    enc = tiktoken.get_encoding("gpt2")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tokens = enc.encode(content)
    data = torch.tensor(tokens, dtype=torch.long)
    
    return data, enc

def analyze_chat_structure(file_path):

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    human_count = content.count('<human>')
    bot_count = content.count('<bot>')
    endoftext_count = content.count('<endOfText>')
    
    print(f"\n=== Data Structure Analysis ===")
    print(f"<human> tags: {human_count}")
    print(f"<bot> tags: {bot_count}")
    print(f"<endOfText> tags: {endoftext_count}")
    

    messages = content.split('<endOfText>')
    
    for i in range(5):
        msg = messages[i].strip()
        print(f"\nMessage {i+1}: {msg}")


def load_all_chat_files(data_dir="data"):
    enc = tiktoken.get_encoding("gpt2")
    all_tokens = []
    
    # get all input files
    input_files = [f"data/input{i}.txt" for i in range(2, 11)]
    print(f"Found {len(input_files)} input files")
    
    total_chars = 0
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tokens = enc.encode(content)
        all_tokens.extend(tokens)
        total_chars += len(content)
    
    print(f"\nTotal characters: {total_chars:,}")
    print(f"Total tokens: {len(all_tokens):,}")
    
    return torch.tensor(all_tokens, dtype=torch.long), enc

if __name__ == "__main__":
    # analyze_chat_structure("data/input4.txt")
    
    all_tokens, tokenizer = load_all_chat_files("data")
    
    '''
    print(f"\nFirst 100 tokens decoded:")
    try:
        print(tokenizer.decode(all_tokens[:100].tolist()))
    except UnicodeEncodeError:
        decoded = tokenizer.decode(all_tokens[:100].tolist())
        print(decoded.encode('ascii', 'ignore').decode('ascii'))
    
    # Save all tokenized data
    '''

    torch.save(all_tokens, "all_chat_tokens.pt")
    print(f"\nAll tokenized data saved to all_chat_tokens.pt")