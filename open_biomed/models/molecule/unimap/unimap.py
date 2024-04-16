import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import copy

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers.activations import gelu

from open_biomed.models.base_models import MolEncoder
from open_biomed.models.molecule.unimap.gcn import DeeperGCN, AtomHead
from open_biomed.models.molecule.unimap.modeling_roberta import RobertaLayer, RobertaModel

def collate_tokens(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config, vocab_size_sp=None, graph_hidden_size=None, atom_vocab_size=None):
        super().__init__()
        first_hidden_size = config.hidden_size
        if graph_hidden_size is not None:
            first_hidden_size += graph_hidden_size
        self.dense = nn.Linear(first_hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if vocab_size_sp is not None:
            vocab_size = vocab_size_sp
        elif atom_vocab_size is not None:
            vocab_size = config.vocab_size + atom_vocab_size
        else:
            vocab_size = config.vocab_size

        self.decoder = nn.Linear(config.hidden_size, vocab_size)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias

class RobertaHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, regression=False):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        output_dim = config.task_output_dim
        self.out_proj = nn.Linear(config.hidden_size, output_dim)
        self.regression = regression

    def forward(self, features, only_feat=False):
        x = self.dropout(features)
        x = self.dense(x)
        if self.regression:
            x = torch.relu(x)
        else:
            x = torch.tanh(x)
        # try diff norm: batch norm, layernorm
        if only_feat:
            return x
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError

class UniMAP(RobertaPreTrainedModel, MolEncoder):
    def __init__(self, multilingua_config, gcn_config, atom_vocab_size, fg_labels_num=85, fingerprint_len=2048, temp=0.05):
        # temp parameter: temperature for the constastive loss
        super(UniMAP, self).__init__(multilingua_config)
        
        self.multilingua_config = multilingua_config
        self.lang_roberta = RobertaModel(multilingua_config, add_pooling_layer=True)
        self.gnn = DeeperGCN(gcn_config)
        
        # mask prediction for the lang model
        # self.lm_head = RobertaLMHead(multilingua_config, atom_vocab_size=atom_vocab_size)

        self.lm_head = RobertaLMHead(multilingua_config)

        #  + atom_vocab_size
        self.lang_gcn_vocab_size = multilingua_config.vocab_size

        # atom mask predcition for the gnn, maybe another RobertaLMHead???
        # self.atom_head = AtomHead(multilingua_config.hidden_size, 
        #                           atom_vocab_size, 'gelu', norm=gcn_config['gnn_norm'])
        
        # todo 1.head for fingerprint regression 2. head function group prediction 3. head for pair matching
        
        self.lang_pooler = Pooler(multilingua_config.pooler_type)
        
        self.config = multilingua_config
        self.gcn_config = gcn_config
        self.loss_mlm = nn.CrossEntropyLoss()
        

        # transfer from gcn embeddings to lang shape
        self.gcn_embedding = nn.Linear(self.gcn_config['gnn_embed_dim'], self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(multilingua_config.hidden_dropout_prob)
        self.LayerNorm = torch.nn.LayerNorm(multilingua_config.hidden_size, eps=1e-12)

        # contrastive head:
        # if smiles_graph: 0, 1; smiles_iupac_graph: 0, 1, 2; 0 means pair match
        contrastive_class_num = multilingua_config.contrastive_class_num
        multilingua_config.task_output_dim = contrastive_class_num
        self.contrastive_head = RobertaHead(multilingua_config)
        self.contrastive_loss = nn.CrossEntropyLoss()
        # self.contrastive_head = nn.Linear(multilingua_config.hidden_size, contrastive_classs_num)
        # function group:
        multilingua_config.task_output_dim = fg_labels_num
        self.fg_task_head = RobertaHead(multilingua_config)
        self.fg_task_loss = nn.BCEWithLogitsLoss(reduction="mean")

        # fingerprint regeression
        multilingua_config.task_output_dim = fingerprint_len
        self.fingerprint_head = RobertaHead(multilingua_config, regression=True)
        self.fingerprint_loss = nn.MSELoss()
        # self.fingerprint_loss = nn.SmoothL1Loss()
        
        # for output token group alignment
        self.lang_group_layer = RobertaLayer(multilingua_config)
        self.sim = Similarity(temp=temp)
        self.ctr_loss = nn.CrossEntropyLoss()
        
        self.output_dim = self.config.hidden_size
    
    
    
    def forward(self, lingua=None, graph=None,):
        return_dict = {}
        if lingua['input_ids'].device != self.device:
            lingua['input_ids'] = lingua['input_ids'].to(self.device)
            if 'mlm_input_ids' in lingua:
                lingua['mlm_input_ids'] = lingua['mlm_input_ids'].to(self.device)
                lingua['mlm_labels'] = lingua['mlm_labels'].to(self.device)
            lingua['attention_mask'] = lingua['attention_mask'].to(self.device)

        graph.to(self.device)
        
        
        gcn_output = self.gnn(graph)
        # concat graph atome embeddings and langua embeddings
        gcn_embedding_output = self.gcn_embedding(gcn_output[1])
        gcn_embedding_output = self.LayerNorm(gcn_embedding_output)
        gcn_embedding_output = self.dropout(gcn_embedding_output)

        assert 'mlm_input_ids' not in lingua
        
        graph_attention_mask = []
        batch_size = lingua['input_ids'].shape[0]
        batch_idx = graph.batch

        gcn_embedding_lst = []
        for bs in range(batch_size):
            gcn_embedding_lst.append(gcn_embedding_output[batch_idx == bs])
            atom_num = (batch_idx == bs).sum().item()
            graph_attention_mask.append(torch.tensor([1 for _ in range(atom_num)]).to(self.device))
        
        graph_attention_mask = collate_tokens(graph_attention_mask, pad_idx=0, pad_to_multiple=8)
        graph_attention_mask = graph_attention_mask.to(torch.bool)

        
        lang_gcn_outputs, lang_gcn_attention_mask = self.lang_roberta(
            lingua['input_ids'],
            attention_mask=lingua['attention_mask'],
            # token_type_ids=lingua['token_type_ids'],
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=True if self.multilingua_config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,

            # graph_input = gcn_embedding_output,
            graph_input = gcn_embedding_lst,
            graph_batch = graph.batch,
            graph_max_seq_size = self.gcn_config['graph_max_seq_size'],
            gnn_mask_labels = None,
            graph_attention_mask = graph_attention_mask,
        )


        last_hidden_embedding = lang_gcn_outputs['last_hidden_state']
        graph_batch = graph.batch
        lang_input_dim = lingua['input_ids'].shape[1]
        lang_gcn_pooler_output = self.lang_pooler(lang_gcn_attention_mask, lang_gcn_outputs)
        if last_hidden_embedding.shape[1] < self.multilingua_config.max_length + self.gcn_config['graph_max_seq_size']:
            bs = last_hidden_embedding.shape[0]
            hidden_size = last_hidden_embedding.shape[2]
            pad_len = self.multilingua_config.max_length + self.gcn_config['graph_max_seq_size'] - last_hidden_embedding.shape[1]
            last_hidden_embedding = torch.cat([
                last_hidden_embedding,
                torch.zeros(bs, pad_len, hidden_size).to(last_hidden_embedding)
            ], dim=1)
            lang_gcn_attention_mask = torch.cat([
                lang_gcn_attention_mask,
                torch.zeros(bs, pad_len).to(lang_gcn_attention_mask)
            ], dim=1)
        
        return lang_gcn_pooler_output, last_hidden_embedding, lang_gcn_attention_mask
        """
        graph_attention_start = lingua['attention_mask'].shape[1]
        out_hidden_embedding = last_hidden_embedding[:, graph_attention_start:, :].contiguous()
        out_attention_mask = lang_gcn_attention_mask[:, graph_attention_start:].contiguous()
        if out_hidden_embedding.shape[1] < self.gcn_config['graph_max_seq_size']:
            bs = out_hidden_embedding.shape[0]
            hidden_size = out_hidden_embedding.shape[2]
            pad_len = self.gcn_config['graph_max_seq_size'] - out_hidden_embedding.shape[1]
            out_hidden_embedding = torch.cat([
                out_hidden_embedding,
                torch.zeros(bs, pad_len, hidden_size).to(out_hidden_embedding)
            ], dim=1)
            out_attention_mask = torch.cat([
                out_attention_mask,
                torch.zeros(bs, pad_len).to(out_attention_mask)
            ], dim=1)
        
        return lang_gcn_pooler_output, out_hidden_embedding, out_attention_mask
        """

    def encode_mol(self, mol):
        h, _, _ = self.forward(mol["smiles"], mol["graph"])
        return h

if __name__ == "__main__":
    config = {
        "data": {
            "mol": {
                "modality": ["structure"],
                "featurizer": {
                    "structure": {
                        "name": "MultiScale",
                        "scales": ["smiles", "graph"],
                        "smiles": {
                            "name": "transformer",
                            "transformer_type": "unimap",
                            "max_length": 128,
                            "model_name_or_path": "../assets/unimap/smiles_tokenizer/"
                        },
                        "graph": {
                            "name": "unimap"
                        }
                    }
                }
            }
        },
        "network": {
            "atom_vocab_size": 10535,
            "roberta": {
                "vocab_size": 2426,
                "max_position_embeddings": 515,
                "type_vocab_size": 1,
                "pooler_type": "avg"
            },
            "gnn": {
                "gnn_number_layer": 3,
                "gnn_dropout": 0.1,
                "conv_encode_edge": True,
                "gnn_embed_dim": 384,
                "gnn_aggr": "maxminmean",
                "gnn_norm": "layer",
                "gnn_act": "gelu",
                "atom_vocab_size": 10535,
                "graph_max_seq_size": 128
            },
        }
    }
    from transformers import RobertaConfig
    from feature.mol_featurizer import MolMultiScaleFeaturizer
    from utils.collators import MolCollator
    roberta_config = RobertaConfig(
        vocab_size=config["network"]["roberta"]["vocab_size"],
        max_position_embeddings=config["network"]["roberta"]["max_position_embeddings"],
        type_vocab_size=config["network"]["roberta"]["type_vocab_size"],
        contrastive_class_num=2,
        pooler_type=config["network"]["roberta"]["pooler_type"]
    )
    featurizer = MolMultiScaleFeaturizer(config["data"]["mol"]["featurizer"]["structure"])
    collator = MolCollator(config["data"]["mol"])
    model = UniMAP(roberta_config, config["network"]["gnn"], config["network"]["atom_vocab_size"])
    model.load_state_dict(torch.load("/share/project/task_3/PUBLIC/Shikun/train_1kw_gcn_3_8gpu_check_frag_final/pytorch_model.bin", map_location="cpu"), strict=False)
    model = model.to("cuda:0")
    model.eval()
    smi1 = "O=C(Nc1ccnc(NC(=O)C2CC2)c1)c1c(Cl)cccc1Cl"
    mol1 = featurizer(smi1)
    smi2 = "CC(C)(C)OC(=O)N[C@H]1CC[C@H](n2nnc3cnc4[nH]ccc4c32)CC1"
    mol2 = featurizer(smi2)
    mol = collator([mol1, mol2])
    print(mol["smiles"], mol["graph"])
    print(model(mol["smiles"], mol["graph"]))
    