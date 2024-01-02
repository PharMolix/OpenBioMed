import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from open_biomed.models.base_models import MolEncoder, TextEncoder
from open_biomed.models.molecule.gnn_graphmvp import GNNGraphMVP
from open_biomed.models.multimodal.molfm.xbert import BertConfig, BertForMaskedLM
from open_biomed.models.knowledge.transe import TransE

from open_biomed.utils.mol_utils import convert_pyg_batch

class MolFM(MolEncoder, TextEncoder):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_n_nodes = config["max_n_nodes"]
        bert_config = BertConfig.from_json_file(config["bert_config_path"])
        self.output_dim = config["gin_hidden_dim"]

        self.structure_encoder = GNNGraphMVP(
            num_layer=config["gin_num_layers"],
            emb_dim=config["gin_hidden_dim"],
            gnn_type="gin",
            drop_ratio=config["drop_ratio"],
            JK="last",
        )
        if "gin_ckpt" in config:
            self.structure_encoder.load_state_dict(torch.load(config["gin_ckpt"]))
        self.structure_proj_head = nn.Linear(config["gin_hidden_dim"], config["projection_dim"])
        self.structure_linear = nn.Linear(config["gin_hidden_dim"], bert_config.hidden_size)
        
        self.text_encoder = BertForMaskedLM(bert_config)
        if "bert_ckpt" in config:
            ckpt = torch.load(config["bert_ckpt"])
            processed_ckpt = {}
            if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in ckpt:
                for k, v in ckpt.items():
                    if k.startswith("module.ptmodel."):
                        processed_ckpt[k[15:]] = v
                    else:
                        processed_ckpt[k] = v
            missing_keys, unexpected_keys = self.text_encoder.load_state_dict(processed_ckpt, strict=False)
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
        self.text_proj_head = nn.Linear(bert_config.hidden_size, config["projection_dim"])
        
        #if "kge" in config:
        #    self.kg_encoder = TransE(**config["kge"])
        #    self.kg_linear = nn.Linear(config["kge"]["hidden_size"], bert_config.hidden_size)

        self.mtm_head = nn.Linear(bert_config.hidden_size, 2)

    def forward(self, mol, text, kg=None, cal_loss=False, output_attentions=False):
        mol_embeds, node_embeds = self.structure_encoder(mol)
        mol_feats = F.normalize(self.structure_proj_head(mol_embeds), dim=-1)
        all_node_feats = self.structure_linear(node_embeds)
        # serialize node feature
        batch_size = mol_feats.shape[0]

        node_feats, node_attention_mask = convert_pyg_batch(all_node_feats, mol.batch, self.max_n_nodes)
        #node_feats = self.structure_linear(mol_embeds).unsqueeze(1)
        #node_attention_mask = torch.ones(batch_size, 1).to(node_feats)

        text_outputs = self.text_encoder.bert(text["input_ids"], attention_mask=text["attention_mask"], mode='text', return_dict=True)
        seq_feats = text_outputs["last_hidden_state"]
        if kg is not None:
            neigh_feats = self.kg_encoder.predict(kg["neigh_indice"])
            neigh_feats = self.kg_linear(neigh_feats)
            node_feats = torch.cat((node_feats, neigh_feats), dim=1)
            node_attention_mask = torch.cat((node_attention_mask, kg["neigh_attn"]), dim=1)
        
        output = self.text_encoder.bert(
            encoder_embeds=seq_feats,
            attention_mask=text["attention_mask"],
            encoder_hidden_states=node_feats,
            encoder_attention_mask=node_attention_mask,
            mode='fusion',
            return_dict=True,
            output_attentions=output_attentions,
        )
        if cal_loss:
            perm =  []
            for i in range(batch_size):
                j = i
                while j == i:
                    j = random.randint(0, batch_size - 1)
                perm.append(j)
            perm = torch.LongTensor(perm).to(seq_feats.device)
            output_neg = self.text_encoder.bert(
                encoder_embeds=seq_feats,
                attention_mask=text["attention_mask"],
                encoder_hidden_states=node_feats[perm],
                encoder_attention_mask=node_attention_mask[perm],
                mode='fusion',
                return_dict=True
            )
            label = torch.cat((torch.ones(batch_size), torch.zeros(batch_size)), dim=0).long().to(seq_feats.device)
            logits = self.mtm_head(torch.cat((output["last_hidden_state"][:, 0, :], output_neg["last_hidden_state"][:, 0, :]), dim=0))
            return F.cross_entropy(logits, label)
        else:
            return output

    def encode_mol(self, structure, proj=False, return_node_feats=False):
        mol_embeds, node_embeds = self.structure_encoder(structure)
        if proj:
            mol_embeds = self.structure_proj_head(mol_embeds)
        if not return_node_feats:
            return mol_embeds
        else:
            return mol_embeds, node_embeds

    def encode_structure_with_prob(self, structure, x, atomic_num_list, device):
        drug_embeds, _ = self.structure_encoder(structure, x, atomic_num_list, device)
        return self.structure_proj_head(drug_embeds) 

    def encode_text(self, text, return_cls=True, proj=True):
        text_embeds = self.text_encoder.bert(text["input_ids"], attention_mask=text["attention_mask"], mode='text', return_dict=True)["last_hidden_state"]
        if return_cls:
            text_embeds = text_embeds[:, 0, :]
        if proj:
            return self.text_proj_head(text_embeds)
        else:
            return text_embeds

    def encode_knowledge(self, kg):
        return self.predict(kg)

    def predict_similarity_score(self, mol, text):
        preds = self.forward(mol, text)["last_hidden_state"][:, 0, :]
        return F.softmax(self.mtm_head(preds), dim=-1)[:, 1]

    def calculate_matching_loss(self, drug, text):
        return self.forward(drug, text, cal_loss=True)