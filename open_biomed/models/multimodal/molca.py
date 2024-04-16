import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, AutoTokenizer, OPTForCausalLM
from peft import PeftModel

from open_biomed.models.base_models import MolEncoder, TextEncoder
from open_biomed.models.molecule.gnn_graphcl import GNNGraphCL
from open_biomed.models.multimodal.molkformer.kformer import BertConfig, BertLMHeadModel
from open_biomed.utils.mol_utils import convert_pyg_batch

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor, mask=None):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class MolCAWrapper(nn.Module):
    def __init__(self, config):
        super(MolCAWrapper, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained(config["bert_tokenizer_path"])
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.graph_encoder = GNNGraphCL(
            num_layer=config["gin_num_layers"],
            emb_dim=config["gin_hidden_dim"],
            gnn_type='gin',
            drop_ratio=config["drop_ratio"],
            JK='last',
        )
        self.ln_graph = LayerNorm(config["gin_hidden_dim"])

        self.qformer_config = BertConfig.from_json_file(config["qformer_config_file"])
        self.num_query_tokens = self.qformer_config.num_query_tokens
        self.Qformer = BertLMHeadModel(self.qformer_config)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.qformer_config.num_query_tokens, self.qformer_config.hidden_size)
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))

    def forward(self, mol, text=None):
        batch_size = torch.max(mol.batch) + 1
        mol_feats, node_feats = self.graph_encoder(mol)
        node_embeds, node_attention_mask = convert_pyg_batch(node_feats, mol.batch, 256)

        query_embeds = self.query_tokens.expand(batch_size, -1, -1)
        if text is None:
            input_ids = None
            attention_mask = torch.ones(query_embeds.shape[:-1], dtype=torch.long).to(query_embeds.device)
        else:
            input_ids = text["input_ids"]
            attention_mask = torch.cat([torch.ones(query_embeds.shape[:-1], dtype=torch.long).to(query_embeds.device), text["attention_mask"]], dim=1)
        return self.Qformer.bert(
            input_ids,
            query_embeds=query_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=node_embeds,
            encoder_attention_mask=node_attention_mask,
            return_dict=True,
        ).last_hidden_state[:, :self.num_query_tokens, :]

class MolCAStage1Wrapper(MolCAWrapper):
    def __init__(self, config):
        super(MolCAStage1Wrapper, self).__init__(config)
        self.graph_proj = nn.Linear(self.Qformer.config.hidden_size, config["projection_dim"])
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, config["projection_dim"])

        self.gtm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

class MolCAStage1(MolEncoder, TextEncoder):
    def __init__(self, config):
        super(MolCAStage1, self).__init__()
        self.blip2qformer = MolCAStage1Wrapper(config)
        self.norm = False

    def forward(self, mol, text):
        return self.blip2qformer(mol, text)

    def encode_mol(self, mol, proj=True):
        repr = self.blip2qformer.forward(mol)
        if proj:
            repr = F.normalize(self.blip2qformer.graph_proj(repr))
        return repr

    def encode_text(self, text, return_cls=True, proj=True):
        text_embeds = self.blip2qformer.Qformer.bert(
            input_ids=text["input_ids"],
            attention_mask=text["attention_mask"],
            return_dict=True
        ).last_hidden_state
        if return_cls:
            text_embeds = text_embeds[:, 0, :]
        if proj:
            text_embeds = F.normalize(self.blip2qformer.text_proj(text_embeds), dim=-1)
        return text_embeds

    def predict_similarity_score(self, mol, text):
        repr = self.blip2qformer.forward(mol, text)
        return self.blip2qformer.gtm_head(repr).mean(dim=1)[:, 1]

class MolCAStage2Wrapper(MolCAWrapper):
    def __init__(self, config):
        super(MolCAStage2Wrapper, self).__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config["text_decoder"], use_fast=False, padding_side='right')
        self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        self.tokenizer.add_tokens('<mol>') # molecule placeholder
        self.mol_token = '<mol>'
        self.tokenizer.mol_token_id = self.tokenizer("<mol>", add_special_tokens=False).input_ids[0]

        self.opt_model = OPTForCausalLM.from_pretrained(config["text_decoder"], torch_dtype=torch.bfloat16)
        self.opt_model = PeftModel.from_pretrained(self.opt_model, config["peft_path"], is_trainable=True)
        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

class MolCAStage2(nn.Module):
    def __init__(self, config) -> None:
        super(MolCAStage2, self).__init__()
        self.blip2opt = MolCAStage2Wrapper(config)

    def decode(self, mol, prompt, num_beams, max_length):
        query_output = self.blip2opt.opt_proj(self.blip2opt(mol))
        inputs_embeds = self.blip2opt.opt.get_input_embeddings()(prompt.input_ids)
        inputs_embeds[prompt.input_ids == self.blip2opt.mol_token_id] = query_output.flatten(0, 1)
        return self.blip2opt.opt_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prompt.attention_mask,
            num_beams=num_beams,
            max_length=max_length,
        )