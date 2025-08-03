import torch

config = {
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'VOCAB': 16384,
    # --- Curriculum Learning Configuration ---
    'CURRICULUM': [64, 96, 128],  # Sequence lengths for each stage
    'MAX_LEN': 128,  # FINAL maximum sequence length
    'EPOCHS_PER_STAGE': 2,  # Epochs to train for each curriculum stage
    # --- Model Hyperparameters ---
    'DIM': 256,
    'LAYERS': 8,
    'HEADS': 4,
    'N_HIER': 2,
    'T_HIER': 2,
    # --- Training Hyperparameters ---
    'BATCH_SIZE': 32,  # Adjusted for potentially smaller sequences
    'LR': 2e-4,
    'NUM_TRAIN_SAMPLES': 100000,
    'TOKENIZER_FILE': "hierarchical_tokenizer.json",
    'ACT_WEIGHT': 0.1
}

final_model_path = 'hierarchical_bert_final.pt'
