import logging
logger = logging.getLogger(__name__)

import argparse
import csv
import collections
import json
import numpy as np
import os
import pickle
import re
from typing import List, Optional

import rdkit.Chem as Chem
from rdkit.Chem import MolStandardize
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import torch
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer

def valid_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi.strip("\n"))
        if mol is not None:
            return True
        else:
            return False
    except:
        return False

def can_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        standardizer = MolStandardize.normalize

        # standardize the molecule
        standardized_mol = standardizer.Normalizer().normalize(mol)
        # get the standardized SMILES string
        standardized_smiles = Chem.MolToSmiles(standardized_mol, isomericSmiles=True)
    except:
        standardized_smiles = smi

    return standardized_smiles

def write_sdf(cid2smiles_file, output_file, sdf_file):
    cid2smiles = pickle.load(open(cid2smiles_file, "rb"))
    smiles2cid = {}
    for cid in cid2smiles:
        if cid2smiles[cid] != '*':
            smi = Chem.MolToSmiles(Chem.MolFromSmiles(cid2smiles[cid]))
            smiles2cid[smi] = cid
    all_mols = []

    print("Loading output file...")
    with open(output_file, "r") as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            line = line.rstrip("\n").split("\t")
            try:
                gt_smi = Chem.MolToSmiles(Chem.MolFromSmiles(line[1]))
                output_mol = Chem.MolFromSmiles(line[2])
                if output_mol is not None:
                    output_mol.SetProp("CID", smiles2cid[gt_smi])
                    all_mols.append(output_mol)
            except:
                continue

    print("Writing sdf file...")
    with Chem.SDWriter(sdf_file) as f:
        for mol in all_mols:
            f.write(mol)

def load_mol2vec(file):
    mol2vec = {}
    with open(file, "r") as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)
        for row in reader:
            mol_str = " ".join(row[-300:])
            mol2vec[row[3]] = np.fromstring(mol_str, sep=" ")
    return mol2vec

def link_datasets(source, target):
    targetsmi2id = {}
    for i, smi in enumerate(target.smiles):
        try:
            smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
            targetsmi2id[smi] = i
        except:
            continue
    match_indexes = []
    for smi in source.smiles:
        try:
            smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
            match_indexes.append(targetsmi2id[smi])
        except:
            match_indexes.append(-1)
    return match_indexes

def convert_pyg_batch(output, batch_idx, max_n_nodes):
    batch_size = torch.max(batch_idx).item() + 1
    batch_output = []
    batch_attention_mask = []
    for i in range(batch_size):
        feat = output[torch.where(batch_idx == i)]
        if feat.shape[0] < max_n_nodes:
            batch_output.append(torch.cat((
                feat,
                torch.zeros(max_n_nodes - feat.shape[0], feat.shape[1]).to(feat.device)
            ), dim=0))
            batch_attention_mask.append(torch.cat((
                torch.ones(feat.shape[0]).to(feat.device), 
                torch.zeros(max_n_nodes - feat.shape[0]).to(feat.device)
            ), dim=0))
        else:
            batch_output.append(feat[:max_n_nodes, :])
            batch_attention_mask.append(torch.ones(max_n_nodes).to(feat.device))
    batch_output = torch.stack(batch_output, dim=0)
    batch_attention_mask = torch.stack(batch_attention_mask, dim=0)
    return batch_output, batch_attention_mask

def add_argument(parser):
    parser.add_argument("--mode", type=str, choices=["write_sdf", "unittest"])

def add_sdf_argument(parser):
    parser.add_argument("--cid2smiles_file", type=str, default="")
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--sdf_file", type=str, default="")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args, _ = parser.parse_known_args()

    if args.mode == "write_sdf":
        add_sdf_argument(parser)
        args = parser.parse_args()
        write_sdf(args.cid2smiles_file, args.output_file, args.sdf_file)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}

def smiles_tokenizer(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return tokens

class SmilesTokenizer(PreTrainedTokenizer):
    """
    Tokenizer in RobertaTokenizer style.
    Creates the SmilesTokenizer class. The tokenizer heavily inherits from the BertTokenizer
    implementation found in Huggingface's transformers library. It runs a WordPiece tokenization
    algorithm over SMILES strings using the tokenisation SMILES regex developed by Schwaller et. al.

    Please see https://github.com/huggingface/transformers
    and https://github.com/rxn4chemistry/rxnfp for more details.

    Examples
    --------
    >>> from deepchem.feat.smiles_tokenizer import SmilesTokenizer
    >>> current_dir = os.path.dirname(os.path.realpath(__file__))
    >>> vocab_path = os.path.join(current_dir, 'tests/data', 'vocab.txt')
    >>> tokenizer = SmilesTokenizer(vocab_path)
    >>> print(tokenizer.encode("CC(=O)OC1=CC=CC=C1C(=O)O"))
    [12, 16, 16, 17, 22, 19, 18, 19, 16, 20, 22, 16, 16, 22, 16, 16, 22, 16, 20, 16, 17, 22, 19, 18, 19, 13]


    References
    ----------
    .. [1] Schwaller, Philippe; Probst, Daniel; Vaucher, Alain C.; Nair, Vishnu H; Kreutter, David;
        Laino, Teodoro; et al. (2019): Mapping the Space of Chemical Reactions using Attention-Based Neural
        Networks. ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv.9897365.v3

    Note
    ----
    This class requires huggingface's transformers and tokenizers libraries to be installed.
    """
    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file: str = '',
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        **kwargs):
        """Constructs a SmilesTokenizer.

        Parameters
        ----------
        vocab_file: str
            Path to a SMILES character per line vocabulary file.
            Default vocab file is found in deepchem/feat/tests/data/vocab.txt
        """

        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        super().__init__(
            vocab_file=vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )


        #super().__init__(vocab_file, **kwargs) #merges_file
        # take into account special tokens in max length
        # self.max_len_single_sentence = self.model_max_length - 2
        # self.max_len_sentences_pair = self.model_max_length - 3

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocab file at path '{}'.".format(vocab_file))
        with open(vocab_file, 'r') as vr:
            self.vocab = json.load(vr)
            # self.vocab = load_vocab(vocab_file)
        # self.highest_unused_index = max(
        #     [i for i, v in enumerate(self.vocab.keys()) if v.startswith("[unused")])
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])

        self.basic_tokenizer = smiles_tokenizer

        self.init_kwargs["model_max_length"] = self.model_max_length

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab_list(self):
        return list(self.vocab.keys())

    def _tokenize(self, text: str):
        """Tokenize a string into a list of tokens.

        Parameters
        ----------
        text: str
            Input string sequence to be tokenized.
        """

        split_tokens = [token for token in self.basic_tokenizer(text)]
        return split_tokens

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A RoBERTa sequence has the following format:

        - single sequence: ``<s> X </s>``
        - pair of sequences: ``<s> A </s></s> B </s>``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def _convert_token_to_id(self, token: str):
        """Converts a token (str/unicode) in an id using the vocab.

        Parameters
        ----------
        token: str
            String token from a larger sequence to be converted to a numerical id.
        """

        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int):
        """Converts an index (integer) in a token (string/unicode) using the vocab.

        Parameters
        ----------
        index: int
            Integer index to be converted back to a string-based token as part of a larger sequence.
        """

        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]):
        """Converts a sequence of tokens (string) in a single string.

        Parameters
        ----------
        tokens: List[str]
            List of tokens for a given string sequence.

        Returns
        -------
        out_string: str
            Single string from combined tokens.
        """

        out_string: str = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def add_special_tokens_ids_single_sequence(self, token_ids: List[int]):
        """Adds special tokens to the a sequence for sequence classification tasks.

        A BERT sequence has the following format: [CLS] X [SEP]

        Parameters
        ----------
        token_ids: list[int]
            list of tokenized input ids. Can be obtained using the encode or encode_plus methods.
        """

        return [self.cls_token_id] + token_ids + [self.sep_token_id]

    def add_special_tokens_single_sequence(self, tokens: List[str]):
        """Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]

        Parameters
        ----------
        tokens: List[str]
            List of tokens for a given string sequence.
        """
        return [self.cls_token] + tokens + [self.sep_token]

    def add_special_tokens_ids_sequence_pair(self, token_ids_0: List[int],
                                            token_ids_1: List[int]) -> List[int]:
        """Adds special tokens to a sequence pair for sequence classification tasks.
        A BERT sequence pair has the following format: [CLS] A [SEP] B [SEP]

        Parameters
        ----------
        token_ids_0: List[int]
            List of ids for the first string sequence in the sequence pair (A).
        token_ids_1: List[int]
            List of tokens for the second string sequence in the sequence pair (B).
        """

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        return cls + token_ids_0 + sep + token_ids_1 + sep

    def add_padding_tokens(self,
                            token_ids: List[int],
                            length: int,
                            right: bool = True) -> List[int]:
        """Adds padding tokens to return a sequence of length max_length.
        By default padding tokens are added to the right of the sequence.

        Parameters
        ----------
        token_ids: list[int]
            list of tokenized input ids. Can be obtained using the encode or encode_plus methods.
        length: int
            TODO
        right: bool, default True
            TODO

        Returns
        -------
        List[int]
            TODO
        """
        padding = [self.pad_token_id] * (length - len(token_ids))

        if right:
            return token_ids + padding
        else:
            return padding + token_ids

    def save_vocabulary(
        self, vocab_path: str
    ):  # -> tuple[str]: doctest issue raised with this return type annotation
        """Save the tokenizer vocabulary to a file.

        Parameters
        ----------
        vocab_path: obj: str
            The directory in which to save the SMILES character per line vocabulary file.
            Default vocab file is found in deepchem/feat/tests/data/vocab.txt

        Returns
        -------
        vocab_file: Tuple
            Paths to the files saved.
            typle with string to a SMILES character per line vocabulary file.
            Default vocab file is found in deepchem/feat/tests/data/vocab.txt
        """
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(
                self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(
                            vocab_file))
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)