# coding=utf-8

import re
import torch
import random
from pathlib import Path
from .util import (DEFAULT_BEGIN_TOKEN, DEFAULT_END_TOKEN, DEFAULT_PAD_TOKEN, \
                  DEFAULT_UNK_TOKEN, DEFAULT_MASK_TOKEN, DEFAULT_SEP_TOKEN, \
                  DEFAULT_MASK_PROB, DEFAULT_SHOW_MASK_TOKEN_PROB, DEFAULT_MASK_SCHEME, \
                  DEFAULT_SPAN_LAMBDA, DEFAULT_VOCAB_PATH, DEFAULT_CHEM_TOKEN_START, REGEX)


class MolEncTokenizer():
    def __init__(
        self,
        vocab,
        chem_token_idxs,
        prog,
        begin_token=DEFAULT_BEGIN_TOKEN,
        end_token=DEFAULT_END_TOKEN,
        pad_token=DEFAULT_PAD_TOKEN,
        unk_token=DEFAULT_UNK_TOKEN,
        mask_token=DEFAULT_MASK_TOKEN,
        sep_token=DEFAULT_SEP_TOKEN,
        mask_prob=DEFAULT_MASK_PROB,
        show_mask_token_prob=DEFAULT_SHOW_MASK_TOKEN_PROB,
        mask_scheme=DEFAULT_MASK_SCHEME,
        span_lambda=DEFAULT_SPAN_LAMBDA
    ):
        """ Initialise the tokenizer

        Args:
            vocab (List[str]): Vocabulary for tokenizer
            chem_token_idxs (List[int]): List of idxs of chemical tokens
            prog (re.Pattern): Regex object for tokenizing
            begin_token (str): Token to use at start of each sequence
            end_token (str): Token to use at end of each sequence
            pad_token (str): Token to use when padding batches of sequences
            unk_token (str): Token to use for tokens which are not in the vocabulary
            mask_token (str): Token to use when masking pieces of the sequence
            sep_token (str): Token to use when sepatating two sentences
            mask_prob (float): Probability of token being masked when masking is enabled
            show_mask_token_prob (float): Probability of a masked token being replaced with mask token
            mask_scheme (str): Masking scheme used by the tokenizer when masking
            span_lambda (float): Mean for poisson distribution when sampling a span of tokens
        """

        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.decode_vocab = {i: t for t, i in self.vocab.items()}
        self.chem_token_idxs = chem_token_idxs
        self.prog = prog

        self.begin_token = begin_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.sep_token = sep_token

        self.mask_prob = mask_prob
        self.show_mask_token_prob = show_mask_token_prob
        self.mask_scheme = mask_scheme
        self.span_lambda = span_lambda

        self.unk_id = self.vocab[unk_token]
        self.unk_token_cnt = {}

    @staticmethod
    def from_vocab_file(
        vocab_path,
        regex,
        chem_tokens_start_idx,
        pad_token_idx=0,
        unk_token_idx=1,
        begin_token_idx=2,
        end_token_idx=3,
        mask_token_idx=4,
        sep_token_idx=5,
        mask_prob=DEFAULT_MASK_PROB,
        show_mask_token_prob=DEFAULT_SHOW_MASK_TOKEN_PROB,
        mask_scheme=DEFAULT_MASK_SCHEME,
        span_lambda=DEFAULT_SPAN_LAMBDA
    ):
        """ Load the tokenizer object from a vocab file and regex

        Reads a newline separated list of tokens from a file to use as the vocabulary
        Note: Assumes that the chemical tokens run from chem_tokens_start_idx to the end of the tokens list
              Anything after the defined tokens and before chem_tokens_start_idx is assumed to be an extra token
              and is added to the regex for tokenizing

        Args:
            vocab_path (str): Path to vocab file
            regex (str): Regex to use for tokenizing
            chem_tokens_start_idx (int): Index of the start of the chemical tokens in the tokens list

        Returns:
            MolEncTokenizer object
        """

        text = Path(vocab_path).read_text()
        tokens = text.split("\n")
        tokens = [t for t in tokens if t is not None and t != ""]

        token_idxs = [pad_token_idx, unk_token_idx, begin_token_idx, end_token_idx, mask_token_idx, sep_token_idx]
        extra_tokens_idxs = range(max(token_idxs) + 1, chem_tokens_start_idx)
        extra_tokens = [tokens[idx] for idx in extra_tokens_idxs]
        prog = MolEncTokenizer._get_compiled_regex(regex, extra_tokens)

        pad_token = tokens[pad_token_idx]
        unk_token = tokens[unk_token_idx]
        begin_token = tokens[begin_token_idx]
        end_token = tokens[end_token_idx]
        mask_token = tokens[mask_token_idx]
        sep_token = tokens[sep_token_idx]

        chem_tokens_idxs = list(range(chem_tokens_start_idx, len(tokens)))
        tokenizer = MolEncTokenizer(
            tokens,
            chem_tokens_idxs,
            prog,
            begin_token=begin_token,
            end_token=end_token,
            pad_token=pad_token,
            unk_token=unk_token,
            mask_token=mask_token,
            sep_token=sep_token,
            mask_prob=mask_prob,
            show_mask_token_prob=show_mask_token_prob,
            mask_scheme=mask_scheme,
            span_lambda=span_lambda
        )
        return tokenizer
    @staticmethod

    def from_pretrained(
        vocab_path,
        regex=REGEX,
        chem_tokens_start_idx=DEFAULT_CHEM_TOKEN_START,
        pad_token_idx=0,
        unk_token_idx=1,
        begin_token_idx=2,
        end_token_idx=3,
        mask_token_idx=4,
        sep_token_idx=5,
        mask_prob=DEFAULT_MASK_PROB,
        show_mask_token_prob=DEFAULT_SHOW_MASK_TOKEN_PROB,
        mask_scheme=DEFAULT_MASK_SCHEME,
        span_lambda=DEFAULT_SPAN_LAMBDA
    ):
        """ Load the tokenizer object from a vocab file and regex

        Reads a newline separated list of tokens from a file to use as the vocabulary
        Note: Assumes that the chemical tokens run from chem_tokens_start_idx to the end of the tokens list
              Anything after the defined tokens and before chem_tokens_start_idx is assumed to be an extra token
              and is added to the regex for tokenizing

        Args:
            vocab_path (str): Path to vocab file
            regex (str): Regex to use for tokenizing
            chem_tokens_start_idx (int): Index of the start of the chemical tokens in the tokens list

        Returns:
            MolEncTokenizer object
        """

        text = Path(vocab_path).read_text()
        tokens = text.split("\n")
        tokens = [t for t in tokens if t is not None and t != ""]

        token_idxs = [pad_token_idx, unk_token_idx, begin_token_idx, end_token_idx, mask_token_idx, sep_token_idx]
        extra_tokens_idxs = range(max(token_idxs) + 1, chem_tokens_start_idx)
        extra_tokens = [tokens[idx] for idx in extra_tokens_idxs]
        prog = MolEncTokenizer._get_compiled_regex(regex, extra_tokens)

        pad_token = tokens[pad_token_idx]
        unk_token = tokens[unk_token_idx]
        begin_token = tokens[begin_token_idx]
        end_token = tokens[end_token_idx]
        mask_token = tokens[mask_token_idx]
        sep_token = tokens[sep_token_idx]

        chem_tokens_idxs = list(range(chem_tokens_start_idx, len(tokens)))
        tokenizer = MolEncTokenizer(
            tokens,
            chem_tokens_idxs,
            prog,
            begin_token=begin_token,
            end_token=end_token,
            pad_token=pad_token,
            unk_token=unk_token,
            mask_token=mask_token,
            sep_token=sep_token,
            mask_prob=mask_prob,
            show_mask_token_prob=show_mask_token_prob,
            mask_scheme=mask_scheme,
            span_lambda=span_lambda
        )
        return tokenizer
    
    @staticmethod
    def from_smiles(
        smiles,
        regex,
        extra_tokens=None,
        begin_token=DEFAULT_BEGIN_TOKEN,
        end_token=DEFAULT_END_TOKEN,
        pad_token=DEFAULT_PAD_TOKEN,
        unk_token=DEFAULT_UNK_TOKEN,
        mask_token=DEFAULT_MASK_TOKEN,
        sep_token=DEFAULT_SEP_TOKEN,
        mask_prob=DEFAULT_MASK_PROB,
        show_mask_token_prob=DEFAULT_SHOW_MASK_TOKEN_PROB,
        mask_scheme=DEFAULT_MASK_SCHEME,
        span_lambda=DEFAULT_SPAN_LAMBDA
    ):
        """ Build the tokenizer from smiles strings and a regex

        Args:
            smiles (List[str]): SMILES strings to use to build vocabulary
            regex (str): Regex to use for tokenizing
            extra_tokens (Optional[List[str]]): Additional tokens to add to the vocabulary that 
                                                may not appear in the SMILES strings
        """

        vocab = {
            pad_token: 0,
            unk_token: 1,
            begin_token: 2,
            end_token: 3,
            mask_token: 4,
            sep_token: 5
        }

        extra_tokens = [] if extra_tokens is None else extra_tokens
        [vocab.setdefault(token, len(vocab)) for token in extra_tokens]

        chem_start_idx = len(vocab)
        prog = MolEncTokenizer._get_compiled_regex(regex, extra_tokens)
        print(f"Chemistry tokens start at index {chem_start_idx}")

        for smi in smiles:
            for token in prog.findall(smi):
                vocab.setdefault(token, len(vocab))

        chem_token_idxs = list(range(chem_start_idx, len(vocab)))

        vocab = sorted(vocab.items(), key=lambda k_v: k_v[1])
        vocab = [key for key, val in vocab]

        tokenizer = MolEncTokenizer(
            vocab,
            chem_token_idxs,
            prog,
            begin_token=begin_token,
            end_token=end_token,
            pad_token=pad_token,
            unk_token=unk_token,
            mask_token=mask_token,
            sep_token=sep_token,
            mask_prob=mask_prob,
            show_mask_token_prob=show_mask_token_prob,
            mask_scheme=mask_scheme,
            span_lambda=span_lambda
        )
        return tokenizer

    def save_vocab(self, vocab_path):
        tokens = sorted(self.vocab.items(), key=lambda k_v: k_v[1])
        tokens = [key for key, val in tokens]

        tokens_str = ""
        for token in tokens:
            tokens_str += f"{token}\n"

        p = Path(vocab_path)
        p.write_text(tokens_str)

    def __len__(self):
        return len(self.vocab)

    def tokenize(self, sents1, sents2=None, mask=False, pad=False):
        if sents2 is not None and len(sents1) != len(sents2):
            raise ValueError("Sentence 1 batch and sentence 2 batch must have the same number of elements")

        tokens = self._regex_match(sents1)
        m_tokens, token_masks = self._mask_tokens(tokens, empty_mask=not mask)

        sent_masks = None
        if sents2 is not None:
            sents2_tokens = self._regex_match(sents2)
            sents2_m_tokens, sents2_masks = self._mask_tokens(sents2_tokens, empty_mask=not mask)
            tokens, sent_masks = self._concat_sentences(tokens, sents2_tokens, self.sep_token)
            m_tokens, _ = self._concat_sentences(m_tokens, sents2_m_tokens, self.sep_token)
            token_masks, _ = self._concat_sentences(token_masks, sents2_masks, False)


        tokens = [[self.begin_token] + ts + [self.end_token] for ts in tokens]
        m_tokens = [[self.begin_token] + ts + [self.end_token] for ts in m_tokens]
        token_masks = [[False] + ts + [False] for ts in token_masks]
        sent_masks = [[0] + mask + [1] for mask in sent_masks] if sent_masks is not None else None

        output = {}

        if pad:
            tokens, orig_pad_masks = self._pad_seqs(tokens, self.pad_token)
            m_tokens, masked_pad_masks = self._pad_seqs(m_tokens, self.pad_token)
            token_masks, _ = self._pad_seqs(token_masks, False)
            sent_masks, _ = self._pad_seqs(sent_masks, False) if sent_masks is not None else (None, None)
            output["original_pad_masks"] = orig_pad_masks
            output["masked_pad_masks"] = masked_pad_masks

        output["original_tokens"] = tokens

        if mask:
            output["masked_tokens"] = m_tokens
            output["token_masks"] = token_masks

        if sent_masks is not None:
            output["sentence_masks"] = sent_masks

        return output

    def _regex_match(self, smiles):
        tokenized = []
        data_type = type(smiles)
        if data_type == str:
            smiles = smiles.split()
        # tokenized = self.prog.findall(smiles)
        for smi in smiles:
            tokens = self.prog.findall(smi)
            tokenized.append(tokens)

        return tokenized

    @staticmethod
    def _get_compiled_regex(regex, extra_tokens):
        regex_string = r"("
        for token in extra_tokens:
            processed_token = token
            for special_character in "()[].|":
                processed_token = processed_token.replace(special_character, f"\\{special_character}")
            regex_string += processed_token + r"|"

        regex_string += regex + r"|"
        regex_string += r".)"
        return re.compile(regex_string)

    def _concat_sentences(self, tokens1, tokens2, sep):
        tokens = [ts1 + [sep] + ts2 for ts1, ts2 in zip(tokens1, tokens2)]
        sent_masks = [([0] * len(ts1)) + [0] + ([1] * len(ts2)) for ts1, ts2 in zip(tokens1, tokens2)]
        return tokens, sent_masks

    def detokenize(self, tokens_list):
        new_tokens_list = []
        for tokens in tokens_list:
            if tokens[0] == self.begin_token:
                tokens = tokens[1:]

            # Remove any tokens after the end token (and end token) if it's there 
            if self.end_token in tokens:
                end_token_idx = tokens.index(self.end_token)
                tokens = tokens[:end_token_idx]

            new_tokens_list.append(tokens)

        strs = ["".join(tokens) for tokens in new_tokens_list]
        return strs

    def convert_tokens_to_ids(self, token_data):
        ids_list = []
        for tokens in token_data:
            for token in tokens:
                token_id = self.vocab.get(token)
                if token_id is None:
                    self._inc_in_dict(self.unk_token_cnt, token)

            ids = [self.vocab.get(token, self.unk_id) for token in tokens]
            ids_list.append(ids)

        return ids_list

    def convert_ids_to_tokens(self, token_ids):
        tokens_list = []
        for ids in token_ids:
            for token_id in ids:
                token = self.decode_vocab.get(token_id)
                if token is None:
                    raise ValueError(f"Token id {token_id} is not recognised")
 
            tokens = [self.decode_vocab.get(token_id) for token_id in ids]
            tokens_list.append(tokens)

        return tokens_list

    def print_unknown_tokens(self):
        print(f"{'Token':<10}Count")
        for token, cnt in self.unk_token_cnt.items():
            print(f"{token:<10}{cnt}")
    
        print()

    @staticmethod
    def _inc_in_dict(coll, item):
        cnt = coll.get(item, 0)
        cnt += 1
        coll[item] = cnt

    def _mask_tokens(self, tokens, empty_mask=False):
        if empty_mask:
            mask = [[False] * len(ts) for ts in tokens]
            return tokens, mask

        masked_tokens = []
        token_masks = []

        for ts in tokens:
            if self.mask_scheme == "replace":
                masked, token_mask = self._mask_replace(ts)
            elif self.mask_scheme == "span":
                masked, token_mask = self._mask_span(ts)
            else:
                raise ValueError(f"Unrecognised mask scheme: {self.mask_scheme}")

            masked_tokens.append(masked)
            token_masks.append(token_mask)

        return masked_tokens, token_masks

    def _mask_replace(self, ts):
        mask_bools = [True, False]
        weights = [self.mask_prob, 1 - self.mask_prob]
        token_mask = random.choices(mask_bools, weights=weights, k=len(ts))
        masked = [self._mask_token(ts[i]) if m else ts[i] for i, m in enumerate(token_mask)]
        return masked, token_mask

    def _mask_span(self, ts):
        curr_token = 0
        masked = []
        token_mask = []

        mask_bools = [True, False]
        weights = [self.mask_prob, 1 - self.mask_prob]
        sampled_mask = random.choices(mask_bools, weights=weights, k=len(ts))

        while curr_token < len(ts):
            # If mask, sample from a poisson dist to get length of mask
            if sampled_mask[curr_token]:
                mask_len = torch.poisson(torch.tensor(self.span_lambda)).long().item()
                masked.append(self.mask_token)
                token_mask.append(True)
                curr_token += mask_len

            # Otherwise don't mask
            else:
                masked.append(ts[curr_token])
                token_mask.append(False)
                curr_token += 1

        return masked, token_mask

    def _mask_token(self, token):
        rand = random.random()
        if rand < self.show_mask_token_prob:
            return self.mask_token

        elif rand < self.show_mask_token_prob + ((1 - self.show_mask_token_prob) / 2):
            token_idx = random.choice(self.chem_token_idxs)
            return self.decode_vocab[token_idx]

        else:
            return token

    @staticmethod
    def _pad_seqs(seqs, pad_token):
        pad_length = max([len(seq) for seq in seqs])
        padded = [seq + ([pad_token] * (pad_length - len(seq))) for seq in seqs]
        masks = [([0] * len(seq)) + ([1] * (pad_length - len(seq))) for seq in seqs]
        return padded, masks


def load_tokenizer(vocab_path=DEFAULT_VOCAB_PATH, chem_token_start=DEFAULT_CHEM_TOKEN_START, regex=REGEX):
    tokenizer = MolEncTokenizer.from_vocab_file(vocab_path, regex, chem_token_start)
    return tokenizer