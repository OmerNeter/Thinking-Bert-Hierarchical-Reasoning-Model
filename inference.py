import torch
import logging
import os
from hierarchical_bert.tokenizer import BPETokenizer
from hierarchical_bert.model import HierarchicalBert

# --- Setup Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Inference Function ---
def run_inference(config, model_path, n_hier_override=None, t_hier_override=None):
    device = config['DEVICE']
    N = n_hier_override if n_hier_override is not None else config['N_HIER']
    T = t_hier_override if t_hier_override is not None else config['T_HIER']

    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}. Please run training first.")
        return

    logger.info(f"\n--- Running Inference with N={N}, T={T} ---")

    tk = BPETokenizer(vocab_size=config['VOCAB'])
    tk.load(config['TOKENIZER_FILE'])

    model = HierarchicalBert(
        tk.vocab_size, config['DIM'], config['LAYERS'], config['HEADS'],
        max_seq=config['MAX_LEN'], enable_act=True
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mask_token_id = tk.special_tokens.get('<MASK>')
    if mask_token_id is None:
        logger.error("Could not find '<MASK>' token in tokenizer.")
        return

    def predict(text_with_mask: str, n_segments: int, t_steps: int):
        logger.info(f"\nQuery: '{text_with_mask}'")
        text_for_encoding = text_with_mask.replace('<MASK>', tk.decode([mask_token_id], skip_special_tokens=False))
        ids = tk.encode(text_for_encoding)

        try:
            mask_idx = ids.index(mask_token_id)
        except ValueError:
            logger.error("Mask token not found in the input text after encoding.")
            return

        L = len(ids)
        pad_len = config['MAX_LEN'] - L
        ids += [tk.special_tokens['<PAD>']] * pad_len
        attn_mask = [1] * L + [0] * pad_len

        inp = torch.tensor([ids], device=device)
        typ = torch.zeros_like(inp)
        msk = torch.tensor([attn_mask], device=device)

        with torch.no_grad():
            logits_list, _ = model(inp, typ, msk, N=n_segments, T=t_steps)

        if not logits_list:
            logger.error("Model did not produce any output logits.")
            return

        final_logits = logits_list[-1]
        final_pred_id = final_logits[0, mask_idx].argmax().item()
        final_pred_token = tk.decode([final_pred_id])
        logger.info(
            f"Final Answer (Segment {len(logits_list)}): '{text_with_mask.replace('<MASK>', final_pred_token)}'")

    predict("The capital of France is <MASK>.", n_segments=N, t_steps=T)
    predict("The large ship sailed across the vast <MASK> for many weeks.", n_segments=N, t_steps=T)
    predict("spaces where art, poetry, and surrealism <MASK> blended to display the anarchist ideal.", n_segments=N,
            t_steps=T)
