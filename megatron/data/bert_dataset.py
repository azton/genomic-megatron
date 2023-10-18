# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""BERT Style dataset."""

import numpy as np
import torch

from megatron import (
    get_args,
    get_tokenizer,
    mpu,
    print_rank_0
)

from megatron.data.gpt_dataset import _build_index_mappings
from megatron.data.dataset_utils import (
    get_samples_mapping,
    get_a_and_b_segments,
    truncate_segments,
    create_tokens_and_tokentypes,
    create_masked_lm_predictions,
    get_datasets_weights_and_num_samples,
)

class BertDataset(torch.utils.data.Dataset):

    def __init__(self, name, indexed_dataset, data_prefix,
                 num_epochs, max_num_samples, masked_lm_prob,
                 max_seq_length, short_seq_prob, seed, binary_head, sliding_window=None):

        # Params to store.
        args = get_args()
        self.genomic = args.genslm or args.genome_k_window
        self.name = name
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob
        self.max_seq_length = max_seq_length
        self.binary_head = binary_head
        self.sliding_window = sliding_window
        # Dataset.
        self.indexed_dataset = indexed_dataset
        # Build index mappings.
    
        # num_samples = len(self.indexed_dataset)
        # total_num_of_documents = indexed_dataset.sizes.shape[0]
        # documents = np.arange(start=0, stop=total_num_of_documents,
        #                 step=1, dtype=np.int32)
                
        # self.doc_idx, self.sample_idx, self.shuffle_idx, self.index_prefix = \
        #     _build_index_mappings(self.name, data_prefix,
        #                           documents, self.indexed_dataset.sizes,
        #                           num_samples, max_seq_length, seed)
        # Build the samples mapping.
        self.samples_mapping = get_samples_mapping(self.indexed_dataset,
                                                   data_prefix,
                                                   num_epochs,
                                                   max_num_samples,
                                                   self.max_seq_length - 3, # account for added tokens
                                                   short_seq_prob,
                                                   self.seed,
                                                   self.name,
                                                   self.binary_head)

        # Vocab stuff.
        tokenizer = get_tokenizer()
        self.vocab_id_list = list(tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_dict = tokenizer.inv_vocab
        self.cls_id = tokenizer.cls
        self.sep_id = tokenizer.sep
        self.mask_id = tokenizer.mask
        self.pad_id = tokenizer.pad

    def __len__(self):
        return self.samples_mapping.shape[0]
        # return self.sample_idx.shape[0] - 1
    def __getitem__(self, idx):
        start_idx, end_idx, seq_length = self.samples_mapping[idx]
        sample = [self.indexed_dataset[i] for i in range(start_idx, end_idx)]
        if self.genomic:
            sample = sample[0]
            if self.sliding_window: # we need to make sure the sample is max_seq_length long
                nleft = (self.max_seq_length - sample.size)
                # print(f"Original sample size: {sample.size}, combining to {self.max_seq_length}: {sample[:10]} ... {sample[-10:]}")
                while nleft > 0:
                    start_idx, end_idx, seq_length = self.samples_mapping[idx+1]
                    ns = self.indexed_dataset[start_idx]
                    # print(f"Adding {ns.size} to sample: {ns[:10]} ... {ns[-10:]}")
                    sample = np.concatenate([sample, ns])
                    nleft = (seq_length - sample.size)
                if sample.size > self.max_seq_length:
                    # print(f"Sample size {sample.size} is greater than max_seq_length {self.max_seq_length}, truncating")
                    sample = sample[:self.max_seq_length-2]
                    # print(f"Truncated sample: {sample[:10]} ... {sample[-10:]}")
            if sample.size < self.max_seq_length: # take a single sample and pad it to max length (GenSLM compatibility)
                sample = np.concatenate([sample, np.zeros(self.max_seq_length - sample.size, dtype=np.int64)+self.pad_id])
        # idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        # doc_index_f = self.sample_idx[idx][0]
        # doc_index_l = self.sample_idx[idx + 1][0]
        # offset_f = self.sample_idx[idx][1]
        # offset_l = self.sample_idx[idx + 1][1]
        # # If we are within the same document, just extract the chunk.
        # doc_ids = []
        # if doc_index_f == doc_index_l:
        #     doc_ids.append(self.doc_idx[doc_index_f])
        #     sample = self.indexed_dataset.get(self.doc_idx[doc_index_f],
        #                                       offset=offset_f,
        #                                       length=offset_l - offset_f + 1)
        # else:
        #     # Otherwise, get the rest of the initial document.
        #     doc_ids.append(self.doc_idx[doc_index_f])
        #     sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f],
        #                                             offset=offset_f)]
        #     # Loop over all in between documents and add the entire document.
        #     for i in range(doc_index_f + 1, doc_index_l):
        #         doc_ids.append(self.doc_idx[i])
        #         sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
        #     # And finally add the relevant portion of last document.
        #     doc_ids.append(self.doc_idx[doc_index_l])
        #     sample_list.append(self.indexed_dataset.get(
        #         self.doc_idx[doc_index_l],
        #         length=offset_l + 1))
        #     sample = np.concatenate(sample_list)
        # sample = sample.reshape(-1)[:self.max_seq_length]
        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        # We % 2**32 since numpy requres the seed to be between 0 and 2**32 - 1
        np_rng = np.random.RandomState(seed=((self.seed + idx) % 2**32))
        if not self.sliding_window:
            return build_training_sample([sample], sample.size,
                                     self.max_seq_length,  # needed for padding
                                     self.vocab_id_list,
                                     self.vocab_id_to_token_dict,
                                     self.cls_id, self.sep_id,
                                     self.mask_id, self.pad_id,
                                     self.masked_lm_prob, np_rng,
                                     self.binary_head)
        else:
            return build_genome_training_sample(sample, sample.size,
                                                self.max_seq_length,  # needed for padding
                                                self.vocab_id_list,
                                                self.vocab_id_to_token_dict,
                                                self.cls_id, self.sep_id,
                                                self.mask_id, self.pad_id,
                                                self.masked_lm_prob, np_rng,
                                                self.sliding_window)

def build_genome_training_sample(sample,
                            target_seq_length, max_seq_length,
                            vocab_id_list, vocab_id_to_token_dict,
                            cls_id, sep_id, mask_id, pad_id,
                            masked_lm_prob, np_rng, sliding_window):
    """ 
        Building a training sample for genomic data
        No N-grams, whole-word masking, etc.
    """
    assert target_seq_length <= max_seq_length
    # print(f"build genome input: {sample}")
    # randomly select masked indices
    masked_inds = np_rng.uniform(size=target_seq_length-2) < masked_lm_prob
    # get the indices of the masked tokens
    masked_positions = np.where(masked_inds)[0]
    sample_input = np.array(sample)
    masked_labels = np.array(sample)
    final_masked_pos = []
    for p in masked_positions:
        # apply mask token 80% of time
        st_mask = p - sliding_window // 2
        end_mask = p + sliding_window // 2 + 1
        if np_rng.uniform() < 0.8:

            sample_input[st_mask:end_mask] = mask_id
            final_masked_pos+=list(range(st_mask,end_mask))

        else:
            if np_rng.uniform() < 0.5:
                # noise with a random token half of remaining time
                for i in range(st_mask,end_mask):
                    sample_input[i] = np_rng.randint(7,71)
                    final_masked_pos.append(i)
            # otherwise, do nothing
            else:
                final_masked_pos.append(p)
    # loss mask is 1 for all masked positions, zero else
    loss_mask_np = np.zeros_like(sample_input)
    loss_mask_np[final_masked_pos] = 1
    # labels should be -1 anywhere that is not a masked position
    labels_np = np.ones_like(sample_input) * -1
    labels_np[final_masked_pos] = masked_labels[final_masked_pos]
    # padding mask is 1 for all positions, zero if padded
    padding_mask_np = np.ones_like(sample_input)
    padding_mask_np[sample_input == pad_id] = 0
    # truncate to target sequence length
    if target_seq_length < max_seq_length:
        truncated = True
    is_next_random = False # only applies to NSP
    tokens_np = sample_input[:target_seq_length]
    labels_np = labels_np[:target_seq_length]
    loss_mask_np = loss_mask_np[:target_seq_length]
    padding_mask_np = padding_mask_np[:target_seq_length]
    # tokentypes is zero for all positions because no NSP
    tokentypes_np = np.zeros_like(tokens_np)
    # print(f"Req masked positions: {masked_positions}")
    # print(f"Built sample: {tokens_np[:10]} ... {tokens_np[-10:]}")
    # print(f"Labels: {labels_np[:10]} ... {labels_np[-10:]}")
    # print(f"Loss mask: {loss_mask_np[:10]} ... {loss_mask_np[-10:]}")
    # print(f"Vocab: {vocab_id_list[:10]} ... {vocab_id_list[-10:]}")

    train_sample = {
        'text': tokens_np.astype(np.int64),
        'types': tokentypes_np.astype(np.int64),
        'labels': labels_np.astype(np.int64),
        'is_random': int(False), #placeholder for compatibility
        'loss_mask': loss_mask_np.astype(np.int64),
        'padding_mask': padding_mask_np.astype(np.int64),
        'truncated': int(False) # same as is_random...
        }
    return train_sample


def build_training_sample(sample,
                          target_seq_length, max_seq_length,
                          vocab_id_list, vocab_id_to_token_dict,
                          cls_id, sep_id, mask_id, pad_id,
                          masked_lm_prob, np_rng, binary_head,):
    """Biuld training sample.

    Arguments:
        sample: A list of sentences in which each sentence is a list token ids.
        target_seq_length: Desired sequence length.
        max_seq_length: Maximum length of the sequence. All values are padded to
            this length.
        vocab_id_list: List of vocabulary ids. Used to pick a random id.
        vocab_id_to_token_dict: A dictionary from vocab ids to text tokens.
        cls_id: Start of example id.
        sep_id: Separator id.
        mask_id: Mask token id.
        pad_id: Padding token id.
        masked_lm_prob: Probability to mask tokens.
        np_rng: Random number genenrator. Note that this rng state should be
              numpy and not python since python randint is inclusive for
              the opper bound whereas the numpy one is exclusive.
    """

    if binary_head:
        # We assume that we have at least two sentences in the sample
        assert len(sample) > 1
    assert target_seq_length <= max_seq_length

    # Divide sample into two segments (A and B).
    if binary_head:
        tokens_a, tokens_b, is_next_random = get_a_and_b_segments(sample,
                                                                  np_rng)
    else:
        tokens_a = []
        for j in range(len(sample)):
            tokens_a.extend(sample[j])
        tokens_b = []
        is_next_random = False

    # Truncate to `target_sequence_length`.
    max_num_tokens = target_seq_length
    print(f"Pre-truncation: {len(tokens_a)} tokens in A, {len(tokens_b)} tokens in B")
    truncated = truncate_segments(tokens_a, tokens_b, len(tokens_a),
                                  len(tokens_b), max_num_tokens, np_rng)
    print(f"Post-truncation: {len(tokens_a)} tokens in A, {len(tokens_b)} tokens in B")

    # Build tokens and toketypes.
    tokens, tokentypes = create_tokens_and_tokentypes(tokens_a, tokens_b,
                                                      cls_id, sep_id)

    # Masking.
    max_predictions_per_seq = masked_lm_prob * max_num_tokens
    (tokens, masked_positions, masked_labels, _, _) = create_masked_lm_predictions(
        tokens, vocab_id_list, vocab_id_to_token_dict, masked_lm_prob,
        cls_id, sep_id, mask_id, max_predictions_per_seq, np_rng)
    print(f"post masked: {len(tokens)} tokens, {len(masked_positions)} masked positions, {len(masked_labels)} masked labels")
    # Padding.
    tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np \
        = pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
                                   masked_labels, pad_id, max_seq_length)
    print(f"post padding: {len(tokens_np)} tokens, {len(tokentypes_np)} tokentypes, {len(labels_np)} labels, {len(padding_mask_np)} padding mask, {len(loss_mask_np)} loss mask")

    train_sample = {
        'text': tokens_np,
        'types': tokentypes_np,
        'labels': labels_np,
        'is_random': int(is_next_random),
        'loss_mask': loss_mask_np,
        'padding_mask': padding_mask_np,
        'truncated': int(truncated)}
    return train_sample


def pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
                             masked_labels, pad_id, max_seq_length):
    """Pad sequences and convert them to numpy."""

    # Some checks.
    num_tokens = len(tokens)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0, \
        f"num_tokens ({num_tokens}) is greater than " \
        "max_seq_length ({max_seq_length})."
    assert len(tokentypes) == num_tokens
    assert len(masked_positions) == len(masked_labels)

    # Tokens and token types.
    filler = [pad_id] * padding_length
    tokens_np = np.array(tokens + filler, dtype=np.int64)
    tokentypes_np = np.array(tokentypes + filler, dtype=np.int64)

    # Padding mask.
    padding_mask_np = np.array([1] * num_tokens + [0] * padding_length,
                               dtype=np.int64)

    # Lables and loss mask.
    labels = [-1] * max_seq_length
    loss_mask = [0] * max_seq_length
    for i in range(len(masked_positions)):
        assert masked_positions[i] < num_tokens
        labels[masked_positions[i]] = masked_labels[i]
        loss_mask[masked_positions[i]] = 1
    labels_np = np.array(labels, dtype=np.int64)
    loss_mask_np = np.array(loss_mask, dtype=np.int64)

    return tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np
