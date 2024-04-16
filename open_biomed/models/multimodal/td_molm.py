import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast

from transformers import BertTokenizer, AutoTokenizer, LlamaForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

from open_biomed.models.base_models import MolEncoder, TextEncoder
from open_biomed.models.molecule.unimol import UniMol
from open_biomed.models.multimodal.molkformer.kformer import BertConfig, BertLMHeadModel
from open_biomed.utils.mol_utils import get_unimol_dictionary

class TDMoLMWrapper(nn.Module):
    def __init__(self, config):
        super(TDMoLMWrapper, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained(config["bert_tokenizer_path"])
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.graph_encoder = UniMol(
            json.load(open(config["structure"]["config_path"], "r")),
            get_unimol_dictionary(config["structure"]["dict_path"]),
        )
        if "ckpt_path" in config["structure"]:
            ckpt = torch.load(config["structure"]["ckpt_path"], map_location="cpu")["model"]
            self.graph_encoder.load_state_dict(ckpt, strict=False)
        self.ln_graph = nn.LayerNorm(self.graph_encoder.num_features)
        

        self.qformer_config = BertConfig.from_json_file(config["qformer_config_file"])
        self.num_query_tokens = self.qformer_config.num_query_tokens
        self.Qformer = BertLMHeadModel(self.qformer_config)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.qformer_config.num_query_tokens, self.qformer_config.hidden_size)
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))

    def _get_structure_features(self, mol):
        node_embeds, node_attention_mask = self.graph_encoder(*mol)
        if node_attention_mask is not None:
            node_attention_mask = 1 - node_attention_mask.long()
        else:
            node_attention_mask = torch.ones(node_embeds.shape[:-1], dtype=torch.long).to(node_embeds.device)
        node_embeds = self.ln_graph(node_embeds)
        return node_embeds, node_attention_mask

    def forward(self, mol, text=None):
        batch_size = mol[0].shape[0]
        node_embeds, node_attention_mask = self._get_structure_features(mol)

        query_embeds = self.query_tokens.expand(batch_size, -1, -1)
        #for name, param in self.Qformer.named_parameters():
        #    print(name, param)
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
            encoder_attention_mask=1 - node_attention_mask,
            return_dict=True,
        ).last_hidden_state[:, :self.num_query_tokens, :]

class TDMoLMStage1Wrapper(TDMoLMWrapper):
    def __init__(self, config):
        super(TDMoLMStage1Wrapper, self).__init__(config)
        self.graph_proj = nn.Linear(self.Qformer.config.hidden_size, config["projection_dim"])
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, config["projection_dim"])
        self.gtm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

class TDMoLMStage1(MolEncoder, TextEncoder):
    def __init__(self, config):
        super(TDMoLMStage1, self).__init__()
        self.blip2qformer = TDMoLMStage1Wrapper(config)
        self.norm = False

    def forward(self, mol, text):
        return self.blip2qformer(mol, text)

    def encode_mol(self, mol, proj=True):
        repr = self.blip2qformer.forward(mol)
        if proj:
            repr = F.normalize(self.blip2qformer.graph_proj(repr), dim=-1)
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

class TDMoLMStage2Wrapper(TDMoLMWrapper):
    def __init__(self, config):
        super(TDMoLMStage2Wrapper, self).__init__(config)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.llm_tokenizer = AutoTokenizer.from_pretrained(config["llm"], use_fast=False, padding_side='right')
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': ['<mol>']})
        self.mol_token_id = self.llm_tokenizer('<mol>', add_special_tokens=False).input_ids[0]

        self.llm_model = LlamaForCausalLM.from_pretrained(config["llm"], torch_dtype=torch.bfloat16)
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        if "peft_model_config" in config:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                r=config["peft_model_config"]["lora_r"],
                lora_alpha=config["peft_model_config"]["lora_alpha"],
                lora_dropout=config["peft_model_config"]["lora_dropout"],
            )
            self.llm_model = get_peft_model(self.llm_model, peft_config)

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

class TDMoLMStage2(nn.Module):
    def __init__(self, config):
        super(TDMoLMStage2, self).__init__()
        self.blip2opt = TDMoLMStage2Wrapper(config)
        self.decoder_tokenizer = self.blip2opt.llm_tokenizer

    @autocast(dtype=torch.bfloat16)
    def decode(self, mol, num_beams, max_length):
        query_output = self.blip2opt.llm_proj(self.blip2opt(mol["conformation"]))
        inputs_embeds = self.blip2opt.llm_model.get_input_embeddings()(mol["SMILES"].input_ids)
        inputs_embeds[mol["SMILES"].input_ids == self.blip2opt.mol_token_id] = query_output.flatten(0, 1)
        return self.blip2opt.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=mol["SMILES"].attention_mask,
            num_beams=num_beams,
            max_length=max_length,
        )

