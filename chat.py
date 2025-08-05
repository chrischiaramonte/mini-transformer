import torch
import tiktoken
from model import BigramLanguageModel, block_size, device

max_new_tokens = 50    
temperature = 0.8      
top_k = 35      

def load_model(model_path='chat_model_best.pt'):
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    
    model = BigramLanguageModel(vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, enc

def generate_response(model, enc, prompt):
    # encode the prompt
    idx = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        generated = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
        full_response = enc.decode(generated[0].tolist())
        
        # extract only the new part
        if len(full_response) > len(prompt):
            response = full_response[len(prompt):]
        else:
            return ""
        
        # stop at end tokens or next human input
        stop_tokens = ['<endOfText>', '<human>']
        for token in stop_tokens:
            if token in response:
                response = response.split(token)[0]
        
        return response.strip()

def chat():
    print("Loading model...")
    
    model, enc = load_model('chat_model_best.pt')
    print("Chat model loaded")
    
    context = ""
    
    print("\nChat started! Type 'quit' to exit.\n")
    
    while True:
        user_input = input("User: ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if not user_input:
            continue
        
        # create prompt for this exchange
        prompt = f"{context}<human>{user_input}<endOfText><bot>"
        
        # generate response
        response = generate_response(model, enc, prompt)
        
        if response:
            print("Bot: " + response)
            # update context with this exchange
            context = f"{context}<human>{user_input}<endOfText><bot>{response}<endOfText>"
        else:
            print("Bot: I'm not sure how to respond to that.")
            context = f"{context}<human>{user_input}<endOfText><bot>I'm not sure how to respond to that.<endOfText>"

if __name__ == "__main__":
    chat()