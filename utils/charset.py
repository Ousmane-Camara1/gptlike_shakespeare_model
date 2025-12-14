import torch
from torch.utils.data import Dataset
import os 

class CharDataset(Dataset): # batch of character sequences

    def __init__(self, config, data_path='/content/drive/MyDrive/Colab Notebooks/mini-project/miniproject2_language_model/data/shakespeare.txt'):
        self.block_size = config.BLOCK_SIZE
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}. Please download it.")
            
        with open(data_path, 'r', encoding='utf-8') as f:
            data = f.read()

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print(f"Data has {data_size} characters, {vocab_size} unique.")
        
        # mappings (string-to-integer, integer-to-string)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = vocab_size

        # converting the entire text to a single tensor of integers
        self.data = torch.tensor([self.stoi[s] for s in data], dtype=torch.long)
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_block_size(self):
        return self.block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        
        # input (x) is the first block_size characters like tokens 0 to N-1
        x = chunk[:-1].clone()
        
        # target (y) is the shifted version (the next block_size characters) so like tokens 1 to N
        y = chunk[1:].clone()
        
        # x is the input, and y is the target (the next token prediction)
        return x, y