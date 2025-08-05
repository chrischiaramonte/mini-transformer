from datasets import load_dataset
import tiktoken
import torch

def load_synthetic_persona_chat():
    print("Loading Synthetic-Persona-Chat dataset...")
    
    # load the dataset
    dataset = load_dataset("google/Synthetic-Persona-Chat", split="train")
    print(f"Loaded {len(dataset)} conversations")
    
    # initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # format conversations for your model
    formatted_conversations = []
    
    for example in dataset:
        # get the conversation text
        conversation_text = example["Best Generated Conversation"]
        
        lines = conversation_text.split('\n')
        formatted_conv = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("User 1:"):
                text = line.replace("User 1:", "").strip()
                formatted_conv += f"<human>{text}<endOfText>\n"
            elif line.startswith("User 2:"):
                text = line.replace("User 2:", "").strip()
                formatted_conv += f"<bot>{text}<endOfText>\n"
        
        if formatted_conv.strip():
            formatted_conversations.append(formatted_conv)
    
    # join all conversations
    full_text = "\n".join(formatted_conversations)
    print(f"Total text length: {len(full_text):,} characters")
    
    # tokenize
    tokens = enc.encode(full_text, allowed_special={'<|endoftext|>'})
    token_tensor = torch.tensor(tokens, dtype=torch.long)
    
    print(f"Total tokens: {len(tokens):,}")
    
    # save tokenized data
    torch.save(token_tensor, "synthetic_persona_tokens.pt")
    print("Saved to synthetic_persona_tokens.pt")
    
    return token_tensor, enc

if __name__ == "__main__":
    tokens, tokenizer = load_synthetic_persona_chat()