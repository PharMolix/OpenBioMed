import logging
logger = logging.getLogger(__name__)

import contextlib

import torch
import torch.nn as nn
import re
import os
from transformers import LlamaTokenizer, EsmModel, EsmConfig

from open_biomed.models.base_models import MolEncoder, ProteinEncoder, TextEncoder
from open_biomed.models.molecule.gnn_graphmvp import GNNGraphMVP
#from models.multimodal.molkformer.mol_kformer import MolKFormer
from open_biomed.models.multimodal.biomedgpt.modeling_llama import LlamaForCausalLM, LlamaConfig
from open_biomed.utils.mol_utils import convert_pyg_batch

class BioMedGPTBase(MolEncoder, ProteinEncoder, TextEncoder):
    def __init__(self):
        super(BioMedGPTBase, self).__init__()

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def from_pretrained(model_name_or_path):
        raise NotImplementedError

    def add_padding(self, wrapped_embeds, wrapped_attention_mask, targets=None, padding="right"):
        batch_size = len(wrapped_embeds)
        max_length_batch = 0
        for i in range(batch_size):
            if wrapped_embeds[i].shape[1] > max_length_batch:
                max_length_batch = wrapped_embeds[i].shape[1]
        for i in range(batch_size):
            if wrapped_embeds[i].shape[1] < max_length_batch:
                pad_len = max_length_batch - wrapped_embeds[i].shape[1]
                if padding == "right":
                    wrapped_embeds[i] = torch.cat((
                        wrapped_embeds[i], 
                        torch.zeros((1, pad_len, wrapped_embeds[i].shape[2]), dtype=wrapped_embeds[i].dtype).to(wrapped_embeds[i].device)
                    ), dim=1)
                    wrapped_attention_mask[i] = torch.cat((
                        wrapped_attention_mask[i],
                        torch.zeros((1, pad_len), dtype=wrapped_attention_mask[i].dtype).to(wrapped_attention_mask[i].device)
                    ), dim=1)
                    if targets is not None:
                        targets[i] = torch.cat((
                            targets[i],
                            torch.ones((1, pad_len), dtype=targets[i].dtype).to(targets[i].device).fill_(-100)
                        ), dim=1)
                else:
                    wrapped_embeds[i] = torch.cat((
                        torch.zeros((1, pad_len, wrapped_embeds[i].shape[2]), dtype=wrapped_embeds[i].dtype).to(wrapped_embeds[i].device),
                        wrapped_embeds[i], 
                    ), dim=1)
                    wrapped_attention_mask[i] = torch.cat((
                        torch.zeros((1, pad_len), dtype=wrapped_attention_mask[i].dtype).to(wrapped_attention_mask[i].device),
                        wrapped_attention_mask[i],
                    ), dim=1)
                    if targets is not None:
                        targets[i] = torch.cat((
                            torch.ones((1, pad_len), dtype=targets[i].dtype).to(targets[i].device).fill_(-100),
                            targets[i],
                        ), dim=1)
        if targets is not None:
            return torch.cat(wrapped_embeds, dim=0), torch.cat(wrapped_attention_mask, dim=0), torch.cat(targets, dim=0)
        else:
            return torch.cat(wrapped_embeds, dim=0), torch.cat(wrapped_attention_mask, dim=0)

    def encode_text(self, text):
        return self.llm(text)

class BioMedGPTV(BioMedGPTBase):
    def __init__(self, config):
        super(BioMedGPTV, self).__init__()
        self.device = config["device"]

        self.mol_structure_config = config["mol"]
        # load molecule structure encoder
        self.mol_structure_encoder = GNNGraphMVP(
            num_layer=self.mol_structure_config["gin_num_layers"],
            emb_dim=self.mol_structure_config["gin_hidden_dim"],
            gnn_type="gin",
            drop_ratio=self.mol_structure_config["drop_ratio"],
            JK="last",
        )
        if config["mol"]["freeze"]:
            logger.info("freeze molecule structure encoder")
            for name, param in self.mol_structure_encoder.named_parameters():
                param.requires_grad = False
            self.mol_structure_encoder = self.mol_structure_encoder.eval()

        # load protein structure encoder
        self.prot_structure_config = EsmConfig.from_json_file(os.path.join(config["protein"]["model_name_or_path"], "config.json"))
        self.prot_structure_encoder = EsmModel(self.prot_structure_config)
        if config["protein"]["use_float16"]:
            self.prot_structure_encoder = self.prot_structure_encoder.half()
        if config["protein"]["lora"]:
            from peft import get_peft_model, LoraConfig, TaskType
            logger.info("applying lora to protein structure encoder")
            lora_config = LoraConfig(peft_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["query", "value"])
            self.prot_structure_encoder = get_peft_model(self.prot_structure_encoder, lora_config)
            self.prot_structure_encoder.print_trainable_parameters()
        elif config["protein"]["freeze"]:
            logger.info("freeze protein structure encoder")
            for name, param in self.prot_structure_encoder.named_parameters():
                param.requires_grad = False
            self.prot_structure_encoder = self.prot_structure_encoder.eval()

        # load llm
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(config["llm"]["model_name_or_path"], use_fast=False, truncation_side="left")
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '<s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '<unk>'})
        logger.info("loading llm")
        self.llm_config = LlamaConfig.from_json_file(os.path.join(config["llm"]["model_name_or_path"], "config.json"))
        self.llm = LlamaForCausalLM(self.llm_config)
        if config["llm"]["use_float16"]:
            self.llm = self.llm.half()
        for name, param in self.llm.named_parameters():
            param.requires_grad = False
        #self.llm = LlamaForCausalLM.from_pretrained(config["llm"]["ckpt"], torch_dtype=torch.float16)
        self.llm.resize_token_embeddings(len(self.llm_tokenizer))

        self.proj_mol = nn.Linear(self.mol_structure_encoder.output_dim, self.llm.config.hidden_size)
        self.proj_prot = nn.Linear(self.prot_structure_encoder.config.hidden_size, self.llm.config.hidden_size)

    def _prompt_wrap(self, mol_feats, mol_batch, prot_feats, cell_feats, text_input, prompt):
        device = text_input.device
        batch_size = mol_feats.shape[0]
        wrapped_embeds_batch, wrapped_attention_mask_batch = [], []
        cur_mol, cur_prot, cur_cell = 0, 0, 0
        for i in range(batch_size):
            text = prompt[i].format(text_input=text_input[i])
            bos_token = torch.ones((1, 1), dtype=text_input.input_ids.dtype, device=text_input.input_ids.device)
            wrapped_embeds = [bos_token * self.llm_tokenizer.bos_token_id]
            pattern = re.compile("<moleculeHere>|<proteinHere>|<cellHere>")
            p_text = pattern.split(text)
            spec_tokens = pattern.findall(text)
            for j in range(len(p_text)):
                p_tokens = self.llm_tokenizer(
                    p_text[j],
                    return_tensors='pt',
                    add_special_tokens=False
                ).to(device)
                p_embeds = self.llm.get_input_embeddings()(p_tokens.input_ids)
                wrapped_embeds.append(p_embeds)
                if j < len(spec_tokens):
                    if spec_tokens[j] == "<moleculeHere>":
                        wrapped_embeds.append(mol_feats.where[mol_batch == cur_mol].unsqueeze(0))
                        cur_mol += 1
                    elif spec_tokens[j] == "<proteinHere>":
                        wrapped_embeds.append(prot_feats[cur_prot].unsqueeze(0))
                        cur_prot += 1
                    elif spec_tokens[j] == "<cellHere>":
                        wrapped_embeds.append(cell_feats[cur_cell].unsqueeze(0))
                        cur_cell += 1
            wrapped_embeds_batch.append(torch.cat(wrapped_embeds, dim=1))
            wrapped_attention_mask_batch.append(torch.ones(wrapped_embeds[-1].shape[:-1]).to(device))
        return wrapped_embeds_batch, wrapped_attention_mask_batch

    def _get_inputs_embeds(self, samples):
        with self.maybe_autocast():
            if "mol" in samples:
                mol_batch = samples["mol"].batch
                _, mol_feats = self.mol_structure_encoder(samples["mol"])
                mol_feats = self.proj_mol(mol_feats)
            else:
                mol_feats, mol_batch = None, None
            
            if "protein" in samples:
                prot_feats = self.prot_structure_encoder(**samples["protein"]).last_hidden_state
                prot_feats = self.proj_prot(prot_feats)

            # TODO: Add cell features
            cell_feats = None

        return self._prompt_wrap(
            mol_feats=mol_feats, 
            mol_batch=mol_batch, 
            prot_feats=prot_feats,
            cell_feats=cell_feats,
            text_input=samples["text_inputs"], 
            prompt=samples["prompt"]
        )

    def forward(self, samples):
        inputs_embeds, inputs_attention_mask = self._get_inputs_embeds(samples)
        
        wrapped_embeds, wrapped_attention_mask, wrapped_targets = [], [], []
        for i in range(len(inputs_embeds)):
            output_tokens = self.llm_tokenizer(
                samples["text_outputs"][i],
                return_tensors='pt',
                add_special_tokens=False
            ).to(inputs_embeds[i].device)
            eos_token = torch.ones((1, 1), dtype=output_tokens.input_ids.dtype, device=output_tokens.input_ids.device)
            output_tokens.input_ids = torch.cat([output_tokens.input_ids, eos_token * self.llm_tokenizer.eos_token_id], dim=1)
            output_tokens.attention_mask = torch.cat([output_tokens.attention_mask, eos_token], dim=1)
            output_embeds = self.llm.get_input_embeddings()(output_tokens.input_ids)
            wrapped_embeds.append(torch.cat([inputs_embeds[i], output_embeds], dim=1))
            wrapped_attention_mask.append(torch.cat([inputs_attention_mask[i], output_tokens.attention_mask], dim=1))
            # do not apply loss to the padding
            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100
            )
            # do not apply loss to the text inputs (i.e., instruction)
            empty_targets = torch.ones(inputs_attention_mask[i].shape, dtype=torch.long).to(inputs_embeds[i].device).fill_(-100)
            wrapped_targets.append(torch.cat([empty_targets, targets], dim=1))
            
        inputs_embeds, inputs_attention_mask, targets = self.add_padding(wrapped_embeds, wrapped_attention_mask, wrapped_targets)
        with self.maybe_autocast():
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs_attention_mask,
                labels=targets,
                return_dict=True
            )
        return outputs.loss

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        with self.maybe_autocast():
            inputs_embeds, inputs_attention_mask = self._get_inputs_embeds(samples)
            inputs_embeds, inputs_attention_mask = self.add_padding(inputs_embeds, inputs_attention_mask)
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs_attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
        
        outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text

    def encode_mol(self, mol, ret_atom_feats=False):
        with self.maybe_autocast():
            mol_feats, node_feats = self.mol_structure_encoder(mol)
            if ret_atom_feats:
                return node_feats
                #if self.mol_structure_config["name"] == "graphmvp":
                #    return convert_pyg_batch(node_feats, mol.batch, self.mol_structure_config["max_n_nodes"])
            else:
                return mol_feats

    def encode_protein(self, protein):
        with self.maybe_autocast():
            return self.prot_structure_encoder(**protein).last_hidden_state

"""
class BioMedGPT2(BioMedGPTBase):
    def __init__(self, config):
        super(BioMedGPT2, self).__init__()
        self.mol_kformer_config = config["kformer"]
        # load molecule kformer
        self.mol_kformer = MolKFormer(self.mol_kformer_config) 
        if "ckpt" in self.mol_kformer_config:
            self.mol_kformer.load_state_dict(torch.load(self.mol_kformer_config["ckpt"], map_location="cpu")["model"], strict=True)

        # load llm
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(config["llm"]["ckpt"], use_fast=False, truncation_side="left")
        #self.llm_tokenizer = GPT2Tokenizer.from_pretrained(llama_ckpt, use_fast=False, truncation_side="left")
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '<s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '<unk>'})
        logger.info("loading llm")
        self.llm = LlamaForCausalLM.from_pretrained(config["llm"]["ckpt"], torch_dtype=torch.float16)
        #self.llm = GPT2LMHeadModel.from_pretrained(llama_ckpt, torch_dtype=torch.float16)
        self.llm.resize_token_embeddings(len(self.llm_tokenizer))
        self.proj = nn.Linear(self.mol_kformer.kqformer_config.hidden_size, self.llm.config.hidden_size)

    def prompt_wrap(self, mol_feats, text_input, prompt):
        batch_size = mol_feats.shape[0]
        wrapped_embeds, wrapped_attention_mask = [], []
        for i in range(batch_size):
            text = prompt[i].format(text_input=text_input[i])
            p_before, p_after = text.split("<moleculeHere>")
            p_before_tokens = self.llm_tokenizer(
                p_before,
                return_tensors='pt',
                add_special_tokens=False,
            ).to(mol_feats.device)
            bos_token = torch.ones((1, 1), dtype=p_before_tokens.input_ids.dtype, device=p_before_tokens.input_ids.device)
            p_before_tokens.input_ids = torch.cat([bos_token * self.llm_tokenizer.bos_token_id, p_before_tokens.input_ids], dim=1)
            p_before_tokens.attention_mask = torch.cat([bos_token, p_before_tokens.attention_mask], dim=1)
            p_after_tokens = self.llm_tokenizer(
                p_after,
                return_tensors='pt',
                add_special_tokens=False
            ).to(mol_feats.device)

            p_before_embeds = self.llm.get_input_embeddings()(p_before_tokens.input_ids)
            p_after_embeds = self.llm.get_input_embeddings()(p_after_tokens.input_ids)
            wrapped_embeds.append(torch.cat([p_before_embeds, mol_feats[i].unsqueeze(0), p_after_embeds], dim=1))
            wrapped_attention_mask.append(torch.ones(wrapped_embeds[-1].shape[:-1]).to(mol_feats.device))
        return wrapped_embeds, wrapped_attention_mask

    def forward(self, samples):
        with self.maybe_autocast():
            text_inputs_kformer = self.mol_kformer.tokenizer(
                samples["text_inputs"],
                return_tensors='pt',
                add_special_tokens=True,
                padding='max_length',
                max_length=self.mol_kformer.max_seq_len
            ).to(samples["mol"].x.device)
            mol_feats = self.mol_kformer(samples["mol"], text_inputs_kformer)
            mol_feats = self.proj(mol_feats)

        inputs_embeds, inputs_attention_mask = self.prompt_wrap(mol_feats, samples["text_inputs"], samples["prompt"])
        
        wrapped_embeds, wrapped_attention_mask, wrapped_targets = [], [], []
        for i in range(len(inputs_embeds)):
            output_tokens = self.llm_tokenizer(
                samples["text_outputs"][i],
                return_tensors='pt',
                add_special_tokens=False
            ).to(inputs_embeds[i].device)
            eos_token = torch.ones((1, 1), dtype=output_tokens.input_ids.dtype, device=output_tokens.input_ids.device)
            output_tokens.input_ids = torch.cat([output_tokens.input_ids, eos_token * self.llm_tokenizer.eos_token_id], dim=1)
            output_tokens.attention_mask = torch.cat([output_tokens.attention_mask, eos_token], dim=1)
            output_embeds = self.llm.get_input_embeddings()(output_tokens.input_ids)
            wrapped_embeds.append(torch.cat([inputs_embeds[i], output_embeds], dim=1))
            wrapped_attention_mask.append(torch.cat([inputs_attention_mask[i], output_tokens.attention_mask], dim=1))
            # do not apply loss to the padding
            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100
            )
            # do not apply loss to the text input (i.e., instruction)
            empty_targets = torch.ones(inputs_attention_mask[i].shape, dtype=torch.long).to(inputs_embeds[i].device).fill_(-100)
            wrapped_targets.append(torch.cat([empty_targets, targets], dim=1))
            
        inputs_embeds, inputs_attention_mask, targets = self.add_padding(wrapped_embeds, wrapped_attention_mask, wrapped_targets)
        with self.maybe_autocast():
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs_attention_mask,
                labels=targets,
                return_dict=True
            )

        return outputs.loss

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        with self.maybe_autocast():
            text_inputs_kformer = self.mol_kformer.tokenizer(
                samples["text_inputs"],
                return_tensors='pt',
                add_special_tokens=True,
                padding='max_length',
                max_length=self.mol_kformer.max_seq_len
            ).to(samples["mol"].x.device)
            mol_feats = self.mol_kformer(samples["mol"], text_inputs_kformer)
            mol_feats = self.proj(mol_feats)

            wrapped_embeds, wrapped_attention_mask = self.prompt_wrap(mol_feats, samples["text_inputs"], samples["prompt"])
            inputs_embeds, attention_mask = self.add_padding(wrapped_embeds, wrapped_attention_mask)
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
        
        outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text

    def encode_mol(self, mol, text=None):
        if text is None:
            text = "No description for the molecule."
        return self.mol_kformer(mol, text)

"""