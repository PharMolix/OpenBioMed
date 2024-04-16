import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaConfig, T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from open_biomed.models.base_models import MolEncoder, TextEncoder
from open_biomed.models.molecule.gnn_graphmvp import GNNGraphMVP
from open_biomed.models.molecule.unimap import UniMAP
from open_biomed.models.molecule.unimol import UniMol
from open_biomed.models.multimodal.molfm.xbert import BertConfig, BertForMaskedLM
from open_biomed.models.knowledge.transe import TransE
from open_biomed.utils.mol_utils import get_unimol_dictionary

class MoELayer(nn.Module):
    def __init__(self, input_dim=256, num_experts=3, k=1):
        super().__init__()
        self.w_gate = nn.Parameter(torch.zeros(input_dim, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_dim, num_experts), requires_grad=True)
        self.k = k
        self.softplus = nn.Softplus()

    def forward(self, input, noise_epsilon=0.2):
        clean_logits = input @ self.w_gate
        if self.train:
            raw_noise_stddev = input @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
        else:
            logits = clean_logits
        _, topk_indices = torch.topk(logits, self.k, dim=-1)
        return topk_indices.squeeze(1)

class Dispatcher(object):
    def __init__(self, num_experts=3) -> None:
        super().__init__()
        self.num_experts = num_experts

    def combine(self, feats, gate_indexes):
        bs = feats[0].shape[0]
        stitched = torch.cat(feats, dim=0)
        indexes = torch.arange(bs).to(gate_indexes) + gate_indexes * bs
        return stitched[indexes]

class DrugFM(MolEncoder, TextEncoder):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_n_atoms = config["max_n_atoms"]
        self.structure_encoders = []
        # GraphMVP
        self.structure_encoders.append(GNNGraphMVP(
            num_layer=config["graphmvp"]["gin_num_layers"],
            emb_dim=config["graphmvp"]["gin_hidden_dim"],
            gnn_type="gin",
            drop_ratio=config["graphmvp"]["drop_ratio"],
            JK="last",
        ))
        roberta_config = RobertaConfig(
            vocab_size=config["unimap"]["roberta"]["vocab_size"],
            max_length=config["unimap"]["roberta"]["max_length"],
            max_position_embeddings=config["unimap"]["roberta"]["max_position_embeddings"],
            type_vocab_size=config["unimap"]["roberta"]["type_vocab_size"],
            contrastive_class_num=2,
            pooler_type=config["unimap"]["roberta"]["pooler_type"]
        )
        # UniMAP
        self.structure_encoders.append(UniMAP(
            roberta_config,
            config["unimap"]["gnn"],
            config["unimap"]["atom_vocab_size"]
        ))
        # UniMol
        self.structure_encoders.append(UniMol(
            json.load(open(config["unimol"]["config_path"], "r")),
            get_unimol_dictionary(config["unimol"]["dict_path"])
        ))
        self.structure_encoders = nn.ModuleList(self.structure_encoders)

        bert_config = BertConfig.from_json_file(config["bert_config_path"])
        self.structure_proj_head, self.structure_linear = [], []
        for i in range(3):
            self.structure_proj_head.append(nn.Linear(self.structure_encoders[i].output_dim, config["projection_dim"]))
            self.structure_linear.append(nn.Linear(self.structure_encoders[i].output_dim, bert_config.hidden_size))
        self.structure_proj_head = nn.ModuleList(self.structure_proj_head)
        self.structure_linear = nn.ModuleList(self.structure_linear)

        self.moe = MoELayer(config["projection_dim"], 3)
        self.dispacher = Dispatcher(3)
        self.text_proj_head = nn.Linear(bert_config.hidden_size, config["projection_dim"])

        self.forward_fns = [self.forward_graphmvp, self.forward_unimap, self.forward_unimol]
        self.model_names = ["graphmvp", "unimap", "unimol"]
        
        # Text encoder
        self.text_encoder = BertForMaskedLM(bert_config)
        self.text_proj_head = nn.Linear(bert_config.hidden_size, config["projection_dim"])

        # Text decoder
        self.text_decoder = T5ForConditionalGeneration.from_pretrained(config["decoder"])
        self.encoder_decoder_proj = nn.Linear(self.text_encoder.config.hidden_size, self.text_decoder.config.hidden_size)
        self.decoder_tokenizer = T5Tokenizer.from_pretrained(config["decoder"])
        
        self.kg_encoder = TransE(**config["kge"])
        self.kg_linear = nn.Linear(config["kge"]["hidden_size"], bert_config.hidden_size)

        self.mtm_head = nn.Linear(bert_config.hidden_size, 2)
        self.norm = False

    def forward_graphmvp(self, mol):
        batch_size = torch.max(mol.batch) + 1
        mol_embeds, node_embeds = self.structure_encoders[0](mol)
        # serialize node feature
        node_feats = []
        node_attention_mask = []
        for i in range(batch_size):
            feat = node_embeds[torch.where(mol.batch == i)]
            if feat.shape[0] < self.max_n_atoms:
                node_feats.append(torch.cat((
                    feat,
                    torch.zeros(self.max_n_atoms - feat.shape[0], feat.shape[1]).to(feat.device)
                ), dim=0))
                node_attention_mask.append(torch.cat((
                    torch.ones(feat.shape[0]).to(feat.device), 
                    torch.zeros(self.max_n_atoms - feat.shape[0]).to(feat.device)
                ), dim=0))
            else:
                node_feats.append(feat[:self.max_n_atoms, :])
                node_attention_mask.append(torch.ones(self.max_n_atoms).to(feat.device))
        node_feats = torch.stack(node_feats, dim=0)
        node_attention_mask = torch.stack(node_attention_mask, dim=0)
        return mol_embeds, node_feats, node_attention_mask

    def forward_unimap(self, mol):
        return self.structure_encoders[1](mol["smiles"], mol["graph"])

    def forward_unimol(self, mol):
        node_embeds, node_attention_mask = self.structure_encoders[2](*mol)
        return node_embeds[:, 0, :], node_embeds, ~node_attention_mask

    def get_mol_text_feats(self, mol, text, kg=None):
        mol_feats, node_feats, node_attention_mask = [], [], []
        for i in range(3):
            cur_mol_feats, cur_node_feats, cur_attention_mask = self.forward_fns[i](mol[self.model_names[i]])
            mol_feats.append(F.normalize(self.structure_proj_head[i](cur_mol_feats)))
            node_feats.append(self.structure_linear[i](cur_node_feats).flatten(1, 2))
            node_attention_mask.append(cur_attention_mask)

        text_outputs = self.text_encoder.bert(text["input_ids"], attention_mask=text["attention_mask"], mode='text', return_dict=True)
        seq_feats = text_outputs["last_hidden_state"]
        text_feats = F.normalize(self.text_proj_head(seq_feats[:, 0, :]), dim=-1)

        expert_ids = self.moe(text_feats)
        mol_feats = self.dispacher.combine(mol_feats, expert_ids)
        node_feats = self.dispacher.combine(node_feats, expert_ids).view(text["input_ids"].shape[0], self.max_n_atoms, -1)
        node_attention_mask = self.dispacher.combine(node_attention_mask, expert_ids)
        if kg is not None:
            neigh_feats = self.kg_encoder.predict(kg["neigh_indice"])
            neigh_feats = self.kg_linear(neigh_feats)
            node_feats = torch.cat((node_feats, neigh_feats), dim=1)
            node_attention_mask = torch.cat((node_attention_mask, kg["neigh_attn"]), dim=1)
        return mol_feats, node_feats, node_attention_mask.long(), seq_feats

    def forward(self, mol, text, kg=None, cal_loss=False, output_attentions=False):
        batch_size = text["input_ids"].shape[0]

        mol_feats, node_feats, node_attention_mask, seq_feats = self.get_mol_text_feats(mol, text, kg)
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
            perm = []
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

    def encode_mol(self, mol, kg=None, proj=True, return_node_feats=False):
        mol_embeds, node_embeds, _, _ = self.get_mol_text_feats(mol["structure"], mol["text"], kg)
        if not return_node_feats:
            return mol_embeds
        else:
            return mol_embeds, node_embeds

    def encode_text(self, text, return_cls=True, proj=True):
        text_embeds = self.text_encoder.bert(text["input_ids"], attention_mask=text["attention_mask"], mode='text', return_dict=True)["last_hidden_state"]
        if return_cls:
            text_embeds = text_embeds[:, 0, :]
        if proj:
            return self.text_proj_head(text_embeds)
        else:
            return text_embeds

    def encode_knowledge(self, kg):
        return self.kg_encoder.predict(kg)

    def predict_similarity_score(self, mol, text):
        preds = self.forward(mol["structure"], text)["last_hidden_state"][:, 0, :]
        return F.softmax(self.mtm_head(preds), dim=-1)[:, 1]

    def calculate_matching_loss(self, mol, text):
        return self.forward(mol, text, cal_loss=True)

    def feat_wrap(self, seq1_feats, seq1_attn, seq2_feats, seq2_attn):
        batch_size = seq1_feats.shape[0]
        wrapped_feats, wrapped_attention_mask = [], []
        for i in range(batch_size):
            cur_len = seq1_attn[i].sum()
            wrapped_feats.append(torch.cat([
                seq1_feats[i, :cur_len],
                seq2_feats[i],
                seq1_feats[i, cur_len:]
            ], dim=0))
            wrapped_attention_mask.append(torch.cat([
                seq1_attn[i, :cur_len],
                seq2_attn[i],
                seq1_attn[i, cur_len:]
            ], dim=0))
        return torch.stack(wrapped_feats, dim=0), torch.stack(wrapped_attention_mask, dim=0)

    def decode(self, mol, num_beams, max_length):
        _, mol_embeds, mol_attention_mask, _ = self.get_mol_text_feats(mol["structure"], mol["text"])
        mol_embeds = self.encoder_decoder_proj(mol_embeds)

        smi_embeds = self.text_decoder.encoder(**mol["structure"]["molt5"]).last_hidden_state
        h, attention_mask = self.feat_wrap(mol_embeds, mol_attention_mask, smi_embeds, mol["structure"]["molt5"].attention_mask)
        h = BaseModelOutput(
            last_hidden_state=h,
            hidden_states=None,
            attentions=None
        )
        outputs = self.text_decoder.generate(
            encoder_outputs=h,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_length=max_length
        )
        return outputs

    def decode_mol(self, text, num_beams, max_length):
        h_text = self.text_decoder.encoder(**text).last_hidden_state
        h = BaseModelOutput(
            last_hidden_state=h_text,
            hidden_states=None,
            attentions=None
        )
        return self.text_decoder.generate(
            encoder_outputs=h,
            attention_mask=text["attention_mask"],
            num_beams=num_beams,
            max_length=max_length
        )

    def causal_generation_loss(self, mol, text):
        labels = text["input_ids"].masked_fill(~text["attention_mask"].bool(), -100)

        _, mol_embeds, mol_attention_mask, _ = self.get_mol_text_feats(mol["structure"], mol["text"])
        mol_embeds = self.encoder_decoder_proj(mol_embeds)

        smi_embeds = self.text_decoder.encoder(**mol["structure"]["molt5"]).last_hidden_state
        h, attention_mask = self.feat_wrap(mol_embeds, mol_attention_mask, smi_embeds, mol["structure"]["molt5"].attention_mask)
        h = BaseModelOutput(
            last_hidden_state=h,
            hidden_states=None,
            attentions=None
        )
        return self.text_decoder(
            encoder_outputs=h,
            attention_mask=attention_mask,
            decoder_attention_mask=text["attention_mask"],
            return_dict=True,
            labels=labels
        ).loss

    def mol_generation_loss(self, text, mol):
        labels = mol["input_ids"].masked_fill(~mol["attention_mask"].bool(), -100)
        h_text = self.text_decoder.encoder(**text).last_hidden_state
        h = BaseModelOutput(
            last_hidden_state=h_text,
            hidden_states=None,
            attentions=None
        )
        return self.text_decoder(
            encoder_outputs=h,
            attention_mask=text["attention_mask"],
            decoder_attention_mask=mol["attention_mask"],
            return_dict=True,
            labels=labels
        ).loss