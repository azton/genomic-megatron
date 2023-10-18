import json
from collections import OrderedDict
from pathlib import Path
from typing import Union, List

PathLike = Union[str, Path]



def load_vocab(filename):
    """
        Expecting a json-formatted vocab file that follows HF conventions
    """
    vocab = OrderedDict()
    with open(filename, 'r') as f:
        tokenizer = json.load(f)
    for key, token in tokenizer['model']['vocab'].items():
        vocab[key] = int(token)
    return vocab

def get_inv_vocab(vocab):
    inv_vocab = OrderedDict()
    for key, token in vocab.items():
        inv_vocab[token] = key
    return inv_vocab

class GenomeTokenizer():
    def __init__(
        self,
        vocab_file: PathLike,
    ) -> None:
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = get_inv_vocab(self.vocab)
        self.k_window = len(list(self.vocab.keys())[7])
        # if self.k_window > 1, we assume a fixed window.
        # TODO: add support for sliding windows; including modification to __getitem__
        self.vocab_size = 71# len(self.vocab)
        self.sliding_window = False
        self.pad_id = self.vocab['[PAD]']
        self.cls_id = self.vocab['[CLS]']
        self.eos_id = self.vocab['[EOS]']
        self.bos_id = self.vocab['[BOS]']
        self.unk_id = self.vocab['[UNK]']
        self.mask_id = self.vocab['[MASK]']
        self.sep_id = self.vocab['[SEP]']

        # bert-specific expectations...
        self.cls = self.cls_id
        self.sep = self.sep_id
        self.mask = self.mask_id
        self.pad = self.pad_id
        self.eod = self.eos_id


    def split_string_by_k(self, seq: str) -> str:
        return ' '.join([seq[i:i+self.k_window] for i in range(0, len(seq)-self.k_window, self.k_window)])


    def encode_sequence(self, seq: str) -> List[int]:
        return [self.vocab[s] if s in self.vocab.keys() else self.unk_id for s in seq.split()]
    
    def decode_sequence(self, seq: List[int]) -> str:
        return ' '.join([self.inv_vocab[s] for s in seq])

    def convert_sliding_to_nt(self, seq: List[str]) -> str:
        # take in a list of tokens and convert to nucleotides
        # for sliding window: act ctg tgc -> actgc 
        if not self.sliding_window:
            seq = ''.join(seq)
        elif self.sliding_window:
            seq = ''.join([s[0] for s in seq[:-1]]) + seq[-1]
        return seq

    def tokenize(self, seq: str) -> str:
        split = self.split_string_by_k(seq)
        encoded = self.encode_sequence(split)
        return encoded

    



