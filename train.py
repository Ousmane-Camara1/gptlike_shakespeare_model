import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import re 
from config import config
from utils.charset import CharDataset
from model.gpt_model import GPTModel

# 2 helper functions

def get_train_val_datasets(data_path, config):
    full_dataset = CharDataset(config, data_path)
    
    # simple train/val split based on indices (did a 0.9/0.1 split 'cause i read online that's better for small datasets)
    train_size = int(config.SPLIT_RATIO * len(full_dataset))
    train_data = full_dataset.data[:train_size]
    val_data = full_dataset.data[train_size:]
    
    # 2 char dataset instances for the split data
    train_dataset = CharDataset(config, data_path, data_tensor=train_data)
    val_dataset = CharDataset(config, data_path, data_tensor=val_data)
    
    vocab_size = full_dataset.get_vocab_size()
    
    return train_dataset, val_dataset, vocab_size, full_dataset.itos # Return itos for generation

def get_data_loaders(dataset, config):
    loader = DataLoader(
        dataset,
        
        # sampler=torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10)), it was not working well with resuming training from checkpoints
        shuffle=True,
        pin_memory=True,
        batch_size=config.BATCH_SIZE,
        num_workers=0 
    )
    return loader


def train():
    print(f"Using device: {config.DEVICE}")
    
    # load data
    full_dataset = CharDataset(config, data_path='/content/drive/MyDrive/Colab Notebooks/mini-project/miniproject2_language_model/data/shakespeare.txt')
    vocab_size = full_dataset.get_vocab_size()
    itos = full_dataset.itos

    # For simplicity and speed, I just trained on the full dataset and relied on the DataLoader to shuffle.
    train_loader = get_data_loaders(full_dataset, config)
    
    # model and optimizer init
    model = GPTModel(config, vocab_size, config.BLOCK_SIZE).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Checkpoint loading and resuming from last step 'cause i kept losing connection in colab due to free tier limits
    start_step = 0
    checkpoint_pattern = re.compile(r'gpt_shakespeare_checkpoint_(\d+)\.pt')
    
    latest_checkpoint = None
    max_step = -1
    for filename in os.listdir('/content/drive/MyDrive/Colab Notebooks/mini-project/miniproject2_language_model/'):
        match = checkpoint_pattern.match(filename)
        if match:
            step_num = int(match.group(1))
            if step_num > max_step:
                max_step = step_num
                latest_checkpoint = filename

    if latest_checkpoint:
        start_step = max_step + 1
        
        # load the entire checkpoint (model + optimizer)
        checkpoint_path = os.path.join('/content/drive/MyDrive/Colab Notebooks/mini-project/miniproject2_language_model/', latest_checkpoint)
        print(f"Loading checkpoint from {checkpoint_path} to resume training from step {start_step}...")
        
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        
        # Assuming the saved file is the full state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])
        # Load the optimizer state which apparently is essential for Adam to resume correctly
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint.get('step', max_step) + 1
        
    # Training Loop
    data_iter = iter(train_loader)
    
    print(f"Starting training from step {start_step} for {config.MAX_STEPS} steps...")
    
    # I used range(start_step, config.MAX_STEPS) to continue from where it left off/crashed
    for step in range(start_step, config.MAX_STEPS): 
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
            
        # Move data to device
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        
        # forward pass, backward pass, Optimization
        model.train()
        _, loss = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()

        # Logging and Checkpoint Saving
        if step % config.LOG_INTERVAL == 0:
            current_loss = loss.item()
            print(f"Step {step}/{config.MAX_STEPS} | Loss: {current_loss:.4f}")

            # inference
            if step > 0:
                
                # checkpoint saving
                checkpoint_path = f'/content/drive/MyDrive/Colab Notebooks/mini-project/miniproject2_language_model/gpt_shakespeare_checkpoint_{step}.pt'
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': current_loss,
                }, checkpoint_path)
                print(f"Model and Optimizer checkpoint saved to {checkpoint_path}")
                
                print("~" * 50)
                print("Generating sample text...")

                context = "O God, O God!"
                context_indices = [full_dataset.stoi[c] for c in context]
                x_context = torch.tensor([context_indices], dtype=torch.long, device=config.DEVICE)
                
                model.eval()
                with torch.no_grad():
                    y_generated = model.generate(
                        x_context, 
                        max_new_tokens=config.GENERATION_LENGTH, 
                        temperature=config.TEMPERATURE
                    )
                
                completion = ''.join([itos[i.item()] for i in y_generated[0]])
                
                print(completion)
                print("~" * 50)
                
    # save model permanently after training
    DRIVE_PATH = '/content/drive/MyDrive/Colab Notebooks/mini-project/miniproject2_language_model/'
    FINAL_PATH = os.path.join(DRIVE_PATH, 'gpt_shakespeare_final.pt')
    print("Training finished. ROGER ROGER saving final model...")
    torch.save(model.state_dict(), FINAL_PATH)


if __name__ == '__main__':
    train()