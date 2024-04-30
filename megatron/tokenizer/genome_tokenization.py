import json
from collections import OrderedDict
from pathlib import Path
from typing import Union, List
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
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
        sliding_window: bool = False,
        single_sample: bool = False,
        BPE: bool = False ) -> None:
        self.printed_outs = 0
        self.vocab = Tokenizer.from_file(vocab_file).get_vocab()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=vocab_file)

        self.inv_vocab = get_inv_vocab(self.vocab)
        self.k_window = len(list(self.vocab.keys())[7])
        self.sliding_window = sliding_window
        self.single_sample = single_sample
        self.bpe_encode = BPE
        self.vocab_size = len(self.vocab)
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
        self.eos = self.eos_id
        self.bos = self.bos_id
    @property
    def eod(self):
        return self.eos
    @property
    def eod_id(self):
        return self.eos_id
    def split_string_by_k(self, seq: str) -> str:
        return ' '.join([seq[i:i+self.k_window] for i in range(0, len(seq)-self.k_window+1, 1 if self.sliding_window else self.k_window)])


    def encode_sequence(self, seq: str) -> List[int]:
        if not self.bpe_encode:
            split = self.split_string_by_k(seq)
            ret = []
            for s in split.split():
                try:
                    ret.append(self.vocab[s.upper()])

                except KeyError as e:
                    try:
                        if 'n' in s.lower() or len(s) < self.k_window:
                            ret.append(self.unk_id)
                    except KeyError as ke:
                        print(f'KeyError: {ke}')
                        print(f'Split: {s}')
                        raise ke
        else:
            raise NotImplementedError
        return ret
    
    def decode_sequence(self, seq: List[int]) -> str:
        return ''.join(self.tokenizer.decode(seq, skip_special_tokens=True))

    # def convert_sliding_to_nt(self, seq: List[str]) -> str:
    #     # take in a list of vocab tokens and convert to nucleotides
    #     # for sliding window: act ctg tgc -> actgc 
    #     if not self.sliding_window:
    #         seq = ''.join(seq)
    #     elif self.sliding_window:
    #         seq = ''.join([s[0] for s in seq[:-1]]) + seq[-1]
    #     return seq

    def tokenize(self, seq: str) -> str:
        if ' ' in seq:
            seq = seq.replace(' ', '')
        if self.bpe_encode:
            # if the sequence is really long, break into chunks
            lseq = range(0, len(seq), 1048576)
            ret_seq = []
            for s in lseq:
                ret_seq += self.tokenizer.encode(seq[s:s+1048576], add_special_tokens=False)
            encoded = ret_seq
            # if self.printed_outs < 2:
            #     print(f"{seq[:100]=}")
            #     print(f"{encoded[:100]=}")
            #     self.printed_outs += 1
        else:
            encoded = self.encode_sequence(seq)
        return encoded


