import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

# Load model from llm_model.py
from llm_model import GPTLanguageModel

# User-configurable training parameters
def get_training_config():
    print("Configure training parameters:")
    batch_size = int(input("Enter batch size (e.g., 32): ") or 32)
    learning_rate = float(input("Enter learning rate (e.g., 0.0003): ") or 0.0003)
    max_epochs = int(input("Enter number of epochs (e.g., 5): ") or 5)
    dataset_path = input("Enter path to text dataset (e.g., 'input.txt'): ") or 'input.txt'
    return {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'max_epochs': max_epochs,
        'dataset_path': dataset_path
    }

# Simple character-level dataset
class TextDataset(Dataset):
    def __init__(self, text, block_size):
        self.block_size = block_size
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]

# Training function
def train_model(model, train_loader, config, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    model.train()
    for epoch in range(config['max_epochs']):
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{config['max_epochs']}, Loss: {total_loss/len(train_loader):.4f}")
    return model

# Decode generated indices to text
def decode(model, dataset, idx):
    return ''.join(dataset.itos[i] for i in idx.tolist())

if __name__ == "__main__":
    # Load model configuration
    model_config = torch.load('model_config.pth')
    
    # Get training configuration
    train_config = get_training_config()
    
    # Load and prepare dataset
    if not os.path.exists(train_config['dataset_path']):
        raise FileNotFoundError(f"Dataset file {train_config['dataset_path']} not found")
    with open(train_config['dataset_path'], 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Initialize dataset and dataloader
    dataset = TextDataset(text, model_config['block_size'])
    train_loader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=True)
    model_config['vocab_size'] = dataset.vocab_size  # Update vocab size from dataset
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTLanguageModel(model_config).to(device)
    print(f"Training on {device} with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Train the model
    model = train_model(model, train_loader, train_config, device)
    
    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')
    
    # Example generation
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=40)
    print("Generated text:", decode(model, dataset, generated[0]))