import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
from datasets import load_dataset
import itertools
import os
from hierarchical_bert.tokenizer import BPETokenizer
from hierarchical_bert.model import HierarchicalBert
from hierarchical_bert.dataset import IterableBertDataset
from torch.utils.data import DataLoader

# --- Setup Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train(config):
    device = config['DEVICE']
    logger.info(f"Using device: {device}")

    # --- Unpack Config ---
    # Model & General Training Params
    VOCAB, MAX_LEN, DIM, LAYERS, HEADS = config['VOCAB'], config['MAX_LEN'], config['DIM'], config['LAYERS'], config[
        'HEADS']
    N_HIER, T_HIER, BATCH_SIZE, LR = config['N_HIER'], config['T_HIER'], config['BATCH_SIZE'], config['LR']
    NUM_TRAIN_SAMPLES, TOKENIZER_FILE, ACT_WEIGHT = config['NUM_TRAIN_SAMPLES'], config['TOKENIZER_FILE'], config[
        'ACT_WEIGHT']

    # Curriculum Learning Params
    CURRICULUM_SCHEDULE = config['CURRICULUM']
    EPOCHS_PER_STAGE = config['EPOCHS_PER_STAGE']

    # --- Tokenizer ---
    tk = BPETokenizer(vocab_size=VOCAB)
    if os.path.exists(TOKENIZER_FILE):
        tk.load(TOKENIZER_FILE)
    else:
        logger.info("Tokenizer file not found, starting training...")
        stream = load_dataset('rumbleFTW/wikipedia-20220301-en-raw', split='train', streaming=True)
        it = (r.get('text', '') for r in itertools.islice(stream, NUM_TRAIN_SAMPLES) if len(r.get('text', '')) >= 10)
        tk.train(it)
        tk.save(TOKENIZER_FILE)

    # --- Model and Optimizer ---
    # Initialize model with the FINAL max sequence length
    model = HierarchicalBert(tk.vocab_size, DIM, LAYERS, HEADS, max_seq=MAX_LEN, enable_act=True).to(device)

    checkpoint_name = 'hierarchical_bert.pt'
    if os.path.exists(checkpoint_name):
        try:
            model.load_state_dict(torch.load(checkpoint_name))
            logger.info(f"Loaded checkpoint from {checkpoint_name}")
        except Exception as e:
            logger.error(f"Could not load checkpoint: {e}. Starting from scratch.")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    # --- Curriculum Training Loop ---
    total_epochs = 0
    for stage, current_max_len in enumerate(CURRICULUM_SCHEDULE):
        logger.info(
            f"--- Curriculum Stage {stage + 1}/{len(CURRICULUM_SCHEDULE)} | Sequence Length: {current_max_len} ---")

        for ep in range(EPOCHS_PER_STAGE):
            total_epochs += 1
            logger.info(f"--- Stage {stage + 1}, Epoch {ep + 1}/{EPOCHS_PER_STAGE} ---")

            # Create a new dataset and dataloader for the current sequence length
            stream = load_dataset('rumbleFTW/wikipedia-20220301-en-raw', split='train', streaming=True)
            shuffled_stream = stream.shuffle(buffer_size=10000, seed=42 + total_epochs)
            it = (r['text'] for r in itertools.islice(shuffled_stream, NUM_TRAIN_SAMPLES) if r.get('text'))
            ds = IterableBertDataset(it, tk, max_len=current_max_len)
            dl = DataLoader(ds, batch_size=BATCH_SIZE)

            model.train()
            pbar = tqdm(dl, desc=f'Stage {stage + 1}, Epoch {ep + 1} (Len: {current_max_len})')

            for batch in pbar:
                inp = batch['input_ids'].to(device)
                typ = batch['token_type_ids'].to(device)
                msk = batch['attention_mask'].to(device)
                lbl = batch['labels'].to(device)

                opt.zero_grad()
                logits_list, q_list = model(inp, typ, msk, N=N_HIER, T=T_HIER)

                total_mlm_loss, total_act_loss = 0.0, 0.0

                for i, logits in enumerate(logits_list):
                    total_mlm_loss += F.cross_entropy(logits.view(-1, model.vocab_size), lbl.view(-1),
                                                      ignore_index=-100)

                    if q_list:
                        with torch.no_grad():
                            reward = ((logits.argmax(-1) == lbl) & (lbl != -100)).float().mean(dim=1)
                            halt_target = reward
                            next_q_max = q_list[i + 1].max(dim=1).values if i < len(q_list) - 1 else torch.zeros_like(
                                reward)
                            targets = torch.stack([halt_target, next_q_max], dim=1)

                        total_act_loss += F.mse_loss(q_list[i], targets)

                avg_mlm_loss = total_mlm_loss / len(logits_list)
                avg_act_loss = total_act_loss / len(q_list) if q_list else torch.tensor(0.0)
                combined_loss = avg_mlm_loss + ACT_WEIGHT * avg_act_loss

                combined_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                opt.step()

                pbar.set_postfix(loss=f'{combined_loss.item():.4f}', mlm=f'{avg_mlm_loss.item():.4f}',
                                 act=f'{avg_act_loss.item():.4f}')

            # Save a checkpoint at the end of each epoch
            path = f'hierarchical_bert_epoch_{total_epochs}.pt'
            torch.save(model.state_dict(), path)
            logger.info(f"Epoch {total_epochs} saved to {path}")

    # Save final model
    final_path = 'hierarchical_bert_final.pt'
    torch.save(model.state_dict(), final_path)
    logger.info(f"Final model saved to {final_path}")