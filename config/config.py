import torch

# Data Parameters
BLOCK_SIZE = 128 # Input & ouput sequence length for the model
BATCH_SIZE = 128  # The number of sequences in a training batch
SPLIT_RATIO = 0.9 

# Model Architecture Parameters
D_MODEL = 768 # embedding dimension (d_model in the "Attention is all you need"paper)
N_LAYERS = 12 # number of transformer blocks (decoder layers)
N_HEADS = 8 # number of attention heads
DROPOUT = 0.1 # during training, 10% of activations are randomly zeroed out
FF_MULTIPLE = 4 # essentially temporarily blow up the representation, apply nonlinearity, then compress it back (Feed-Forward Network hidden layer (D_MODEL * 4))
 
# Training Parameters
LEARNING_RATE = 3e-4 # for the Adam optimizer
MAX_STEPS = 6000 
LOG_INTERVAL = 500
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# generation parameters
GENERATION_LENGTH = 500 
TEMPERATURE = 1.0 # use the model’s probabilities as-is, no extra confidence or chaos.