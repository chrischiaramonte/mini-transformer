import torch
import tiktoken
from model import BigramLanguageModel, block_size, device

# settings
max_new_tokens = 150
temperature = 0.8

def load_model(model_path='chat_model.pt'):
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    
    model = BigramLanguageModel(vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, enc

def generate_response(model, enc, prompt, max_new_tokens=max_new_tokens, temperature=temperature):
    # predicts from this
    context = f"<human>{prompt}<endOfText><bot>"
    
    idx = torch.tensor(enc.encode(context), dtype=torch.long).unsqueeze(0).to(device)
    
    # generate
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
                
    response = enc.decode(idx[0].tolist())
    
    if '<bot>' in response:
        response = response.split('<bot>')[-1]
        if '<endOfText>' in response:
            response = response.split('<endOfText>')[0]
        return response.strip()
    
    return response

def chat():
    print("Loading model...")
 
    model, enc = load_model('chat_model.pt')
    print("Chat model loaded")
   
    
    print("\nChat started! Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if not user_input:
            continue
        
        # generate response
        print("Bot: ", end="", flush=True)
        response = generate_response(model, enc, user_input)
        print(response)
        print()

if __name__ == "__main__":
    chat()