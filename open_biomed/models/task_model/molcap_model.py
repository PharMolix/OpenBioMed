import torch
import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutput

from open_biomed.models import SUPPORTED_MOL_ENCODER, SUPPORTED_TEXT_DECODER

from open_biomed.utils.mol_utils import convert_pyg_batch

class MolCapModel(nn.Module):
    def __init__(self, config):
        super(MolCapModel, self).__init__()
        self.generate_model = SUPPORTED_TEXT_DECODER[config["text"]["name"]](config["text"])
        if "structure" in config:
            self.encode_model = SUPPORTED_MOL_ENCODER[config["structure"]["name"]](**config["structure"])
            self.max_n_nodes = config["structure"]["max_n_nodes"]
        else:
            self.encode_model = None
        self.decoder_tokenizer = self.generate_model.decoder_tokenizer

    def forward(self, mol, text):
        labels = text["input_ids"].masked_fill(~text["attention_mask"].bool(), -100)
        h, attention_mask = self.encode(mol)
        return self.generate_model(
            encoder_outputs=h,
            attention_mask=attention_mask,
            decoder_attention_mask=text["attention_mask"],
            labels=labels
        )

    def encode(self, mol):
        if self.encode_model is None:
            h = self.generate_model.encode_mol(mol)
            encoder_attention_mask = mol["attention_mask"]
        else:
            _, node_feats = self.encode_model.encode_mol(mol)
            h, encoder_attention_mask = convert_pyg_batch(node_feats, mol.batch, max_n_nodes=self.max_n_nodes)
        h = BaseModelOutput(
            last_hidden_state=h,
            hidden_states=None,
            attentions=None
        )
        return h, encoder_attention_mask

    def decode(self, mol, num_beams, max_length):
        h, attention_mask = self.encode(mol)
        return self.generate_model.decode(
            encoder_outputs=h,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_length=max_length
        )

    def causal_generation_loss(self, mol, text):
        return self.forward(mol, text)

class GraphEnhancedMolCapModel(nn.Module):
    def __init__(self, config):
        super(GraphEnhancedMolCapModel, self).__init__()
        self.generate_model = SUPPORTED_TEXT_DECODER[config["text"]["name"]](config["text"])
        self.graph_encoder = SUPPORTED_MOL_ENCODER[config["graph"]["name"]](config["graph"])
        if "init_checkpoint" in config["graph"]:
            ckpt = torch.load(config["graph"]["init_checkpoint"])
            if "param_key" in config["graph"]:
                ckpt = ckpt[config["graph"]["param_key"]]
            self.graph_encoder.load_state_dict(ckpt)
        if config["graph"]["stop_grad"]:
            for k, v in self.graph_encoder.named_parameters():
                v.requires_grad = False
        self.graph_projector = nn.Sequential(
            nn.Linear(config["graph"]["output_dim"], self.generate_model.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.generate_model.hidden_size, self.generate_model.hidden_size)
        )

        self.max_n_nodes = config["graph"]["max_n_nodes"]
        self.use_node_embeds = config["graph"]["max_n_nodes"] > 0
        self.decoder_tokenizer = self.generate_model.decoder_tokenizer

    def forward(self, mol, text):
        h, attention_mask = self.encode(mol)
        labels = text["input_ids"].masked_fill(~text["attention_mask"].bool(), -100)
        return self.generate_model(
            encoder_outputs=h,
            attention_mask=attention_mask,
            decoder_attention_mask=text["attention_mask"],
            labels=labels
        )

    def decode(self, mol, num_beams, max_length):
        h, attention_mask = self.encode(mol)
        return self.generate_model.decode(
            encoder_outputs=h,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_length=max_length
        )

    def encode(self, mol):
        B, _ = mol["SMILES"]["attention_mask"].shape
        device = mol["SMILES"]["attention_mask"].device
        smi_feats = self.generate_model.encode_mol(mol["SMILES"])
        if self.use_node_embeds:
            graph_feats, node_feats = self.graph_encoder.encode_mol(mol["graph"], proj=False, return_node_feats=True)
            graph_feats = self.graph_projector(graph_feats)
            node_feats, node_attention_mask = convert_pyg_batch(node_feats, mol["graph"].batch, self.max_n_nodes)
            node_feats = self.graph_projector(node_feats)
            h = BaseModelOutput(
                last_hidden_state=torch.cat([graph_feats.unsqueeze(1), node_feats, smi_feats], dim=1),
                #last_hidden_state=torch.cat([graph_feats.unsqueeze(1), node_feats], dim=1),
                hidden_states=None,
                attentions=None
            )
            encoder_attention_mask = torch.cat([torch.ones(B, 1).to(device), node_attention_mask, mol["SMILES"]["attention_mask"]], dim=1)
            #encoder_attention_mask = torch.cat([torch.ones(B, 1).to(device), node_attention_mask], dim=1)
            
        return h, encoder_attention_mask

    def causal_generation_loss(self, mol, text):
        return self.forward(mol, text)