import torch
import torch.nn as nn
import os
import argparse

from config import config
from utils.charset import CharDataset
from model import gpt_model

# my final model
MODEL_PATH = 'gpt_shakespeare_final.pt' 

DATA_PATH = './data/shakespeare.txt'


def load_model_and_data(model_path):

    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at expected local path: {DATA_PATH}")
        print("Remember you ain't in kansas anymore (kansas being colab)")
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}.")
        
    full_dataset = CharDataset(config, data_path=DATA_PATH) 
    vocab_size = full_dataset.get_vocab_size()
    
    # init model
    model = gpt_model.GPTModel(config, vocab_size, config.BLOCK_SIZE).to(config.DEVICE)
    
    # load weights
    if not os.path.exists(model_path):
        print(f"Model weights not found at expected local path: {model_path}")
        print("Remember you ain't in kansas anymore (kansas being colab).")
        raise FileNotFoundError(f"Model weights not found at {model_path}.")
        
    print(f"Loading trained weights from: {model_path}")
    
    # load state dict and map to the specified device (I tried running it on pc, on mac and in colab but ended up with colab cause 'cuda')
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval() # setting model to evaluation mode for inference
    
    return model, full_dataset


def generate_text(model, dataset, seed_text, max_new_tokens, temperature):
    # tokenize the context
    try:
        context_indices = [dataset.stoi[c] for c in seed_text]
    except KeyError as e:
        print(f"oh oooh: what manner of sorcellerie is this: {e}.")
        return ""
        
    # convert list of indices to a tensor (1 batch, T tokens)
    x_context = torch.tensor([context_indices], dtype=torch.long, device=config.DEVICE)
    
    # generate new tokens
    with torch.no_grad():
        y_generated = model.generate(
            x_context, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature
        )
    
    # decode tokens back to string
    completion = ''.join([dataset.itos[i.item()] for i in y_generated[0]])
    
    return completion


def main():
    
    # just to allow myself to test different parameters from command line
    parser = argparse.ArgumentParser(description="Generate Shakespearean text using a trained GPT model.")
    parser.add_argument('--seed', type=str, default="O God, O God!", 
                        help="Initial text prompt for generation.")
    parser.add_argument('--length', type=int, default=config.GENERATION_LENGTH, 
                        help="Number of tokens to generate.")
    parser.add_argument('--temp', type=float, default=config.TEMPERATURE, 
                        help="Sampling temperature (lower=deterministic, higher=creative).")
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, 
                        help="Path to the trained model weights (.pt file).")
    
    args = parser.parse_args()

    try:
        model, full_dataset = load_model_and_data(args.model_path)
        
        print("\n" + "="*70)
        print(f"Starting Generation on Device: {config.DEVICE}")
        print(f"Model Configuration: D_MODEL={config.D_MODEL}, N_LAYERS={config.N_LAYERS}")
        print("="*70 + "\n")
        
        print(f"--- Prompt: '{args.seed}'")
        print(f"--- Length: {args.length}, Temperature: {args.temp}\n")
        
        generated_text = generate_text(
            model, 
            full_dataset, 
            args.seed, 
            args.length, 
            args.temp
        )
        
        print(generated_text)
        print("\n" + "-"*70)
        print("Roger Roger generation complete.")

    except FileNotFoundError as e:
        pass 
    except Exception as e:
        print(f"\n[OH OOOH, ME LORD UNEXPECTED ERROR]: {e}")
        
if __name__ == '__main__':
    main()