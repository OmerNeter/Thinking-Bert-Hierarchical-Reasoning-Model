import torch
from typing import List, Iterator, Dict, Union
import logging
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# --- Setup Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Tokenizer ---
class BPETokenizer:
    def __init__(self, vocab_size: int = 32768):
        self.vocab_size_target = vocab_size
        self.special_token_list = ['<PAD>', '<UNK>', '<CLS>', '<SEP>', '<MASK>']
        self.tokenizer = Tokenizer(BPE(unk_token='<UNK>'))
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()
        self._update_special_tokens_map()

    def train(self, texts: Iterator[str]):
        logger.info('Training BPE tokenizer...')
        trainer = BpeTrainer(vocab_size=self.vocab_size_target, special_tokens=self.special_token_list)
        self.tokenizer.train_from_iterator(texts, trainer)
        self._update_special_tokens_map()
        logger.info(f'Tokenizer vocab size: {self.vocab_size}')

    def _update_special_tokens_map(self):
        self._special_tokens_map = {
            tok: self.tokenizer.token_to_id(tok) for tok in self.special_token_list if
            self.tokenizer.token_to_id(tok) is not None
        }

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, tokens: Union[torch.Tensor, List[int]], skip_special_tokens=True) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().tolist()
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def save(self, path: str):
        self.tokenizer.save(path)
        logger.info(f'Tokenizer saved to {path}')

    def load(self, path: str):
        self.tokenizer = Tokenizer.from_file(path)
        self._update_special_tokens_map()
        logger.info(f'Tokenizer loaded from {path}')

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def special_tokens(self) -> Dict[str, int]:
        return self._special_tokens_map
