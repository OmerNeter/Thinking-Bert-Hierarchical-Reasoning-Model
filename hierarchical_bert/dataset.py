import torch
from torch.utils.data import IterableDataset
from typing import Iterator
import random
from tokenizer import BPETokenizer


class IterableBertDataset(IterableDataset):
    def __init__(self, texts: Iterator[str], tokenizer: BPETokenizer, max_len: int = 128):
        self.texts = texts
        self.tk = tokenizer
        self.max_len = max_len  # This will be set dynamically during curriculum training

    def __iter__(self):
        for txt in self.texts:
            if not txt or len(txt.split()) < 5: continue
            ids = self.tk.encode(txt)[:self.max_len - 2]
            ids = [self.tk.special_tokens['<CLS>']] + ids + [self.tk.special_tokens['<SEP>']]
            L = len(ids)
            labels = [-100] * L

            num_to_mask = max(1, int((L - 2) * 0.15))
            if L - 2 < num_to_mask: continue

            mask_indices = random.sample(range(1, L - 1), num_to_mask)
            for i in mask_indices:
                labels[i] = ids[i]
                r = random.random()
                if r < 0.8:
                    ids[i] = self.tk.special_tokens['<MASK>']
                elif r < 0.9:
                    rand_token_id = random.randint(len(self.tk.special_tokens), self.tk.vocab_size - 1)
                    ids[i] = rand_token_id

            pad_len = self.max_len - L
            ids += [self.tk.special_tokens['<PAD>']] * pad_len
            labels += [-100] * pad_len
            attention_mask = [1] * L + [0] * pad_len

            yield {
                'input_ids': torch.tensor(ids, dtype=torch.long),
                'token_type_ids': torch.zeros(self.max_len, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long)
            }
