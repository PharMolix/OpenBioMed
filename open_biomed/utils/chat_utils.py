import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import logging
logger = logging.getLogger(__name__)

import re

from utils.data_utils import DataProcessorFast
from utils.mol_utils import valid_smiles
from utils.collators import ToDevice

class Conversation(object):
    def __init__(self, model, processor_config, device, system, roles=("Human", "Assistant"), sep="###", max_length=512):
        self.model = model
        self.mol_processor = DataProcessorFast("molecule", processor_config["mol"])
        self.prot_processor = DataProcessorFast("protein", processor_config["protein"])
        #TODO: add cell
        self.device = device
        self.system = system
        self.roles = roles
        self.sep = sep
        self.max_length = max_length
        self.messages = []
        self.mol_embs = []
        self.prot_embs = []
        self.cell_embs = []

    def _wrap_prompt(self):
        ret = self.system + self.sep + " "
        for role, message in self.messages:
            if message:
                ret += self.roles[role] + ": " + message + " " + self.sep + " "
            else:
                ret += self.roles[role] + ": "
        return ret

    def _append_message(self, role, message):
        self.messages.append([role, message])

    def _get_context_emb(self):
        prompt = self._wrap_prompt()
        logger.debug("Prompt: %s" % (prompt))
        pattern = re.compile("<moleculeHere>|<proteinHere>|<cellHere>")
        p_text = pattern.split(prompt)
        spec_tokens = pattern.findall(prompt)
        assert len(p_text) == len(self.mol_embs) + len(self.prot_embs) + len(self.cell_embs) + 1, "Unmatched numbers of placeholders and molecules."
        seg_tokens = [
            self.model.llm_tokenizer([seg], return_tensors="pt", add_special_tokens=(i == 0)).to(self.device)
            for i, seg in enumerate(p_text) 
        ]
        seg_embs = [self.model.llm.get_input_embeddings()(seg_token.input_ids) for seg_token in seg_tokens]
        mixed_embs = []
        cur_mol, cur_prot, cur_cell = 0, 0, 0
        for i in range(len(p_text) - 1):
            mixed_embs.append(seg_embs[i])
            if spec_tokens[i] == "<moleculeHere>":
                mixed_embs.append(self.mol_embs[cur_mol])
                cur_mol += 1
            elif spec_tokens[i] == "<proteinHere>":
                mixed_embs.append(self.prot_embs[cur_prot])
                cur_prot += 1
            elif spec_tokens[i] == "<cellHere>":
                mixed_embs.append(self.cell_embs[cur_cell])
                cur_cell += 1
        mixed_embs.append(seg_embs[-1])
        return torch.cat(mixed_embs, dim=1)

    def ask(self, text):
        if len(self.messages) > 0 and (self.messages[-1][1].endswith("</molecule>") or self.messages[-1][1].endswith("</protein>")) and self.messages[-1][0] == 0:
            self.messages[-1][1] = self.messages[-1][1] + " " + text
        else:
            self._append_message(0, text)
    
    def append_molecule(self, smi):
        if not valid_smiles(smi):
            logger.error("Failed to generate molecule graph. Maybe the SMILES is invalid.")
            return
        mol_inputs = ToDevice(self.mol_processor(smi), self.device)
        with self.model.maybe_autocast():
            mol_embs = self.model.proj_mol(self.model.encode_mol(mol_inputs, ret_atom_feats=True))
        self.mol_embs.append(mol_embs.unsqueeze(0))
        self._append_message(0, "<molecule><moleculeHere></molecule>")

    def append_protein(self, protein, from_file=False):
        if from_file:
            protein = open(protein, "r").readline()
        prot_inputs = ToDevice(self.prot_processor(protein), self.device)
        with self.model.maybe_autocast():
            prot_embs = self.model.proj_prot(self.model.encode_protein(prot_inputs))
        self.prot_embs.append(prot_embs)
        self._append_message(0, "<protein><proteinHere></protein>")

    def answer(self, max_new_tokens=256, num_beams=1, top_p=0.9, repetition_penalty=1.0, length_penalty=1, temperature=1.0,):
        self._append_message(1, None)
        embs = self._get_context_emb()

        if embs.shape[1] + max_new_tokens > self.max_length:
            begin_idx = embs.shape[1] + max_new_tokens - self.max_length
            embs = embs[:, begin_idx]
            logger.warn("The number of tokens in current conversation exceeds the max length (%d). The model will not see the contexts outside the range." % (self.max_length))
        
        output = self.model.llm.generate(
            inputs_embeds=embs,
            max_length=max_new_tokens,
            num_beams=num_beams,
            top_p=top_p,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature
        )[0]
        if output[0] in [0, 1]:
            output = output[1:]
        output = output[:-1]
        output_tokens = self.model.llm_tokenizer.decode(output, add_special_tokens=False)
        output_tokens = output_tokens.split("Assistant:")[-1].strip()
        self.messages[-1][1] = output_tokens
        return output_tokens, output.cpu().numpy()

    def reset(self):
        self.messages = []
        self.mol_embs = []
        self.prot_embs = []
        self.cell_embs = []
    

if __name__ == "__main__":
    import json
    from models.multimodal import BioMedGPTV

    config = json.load(open("./configs/encoders/multimodal/biomedgptv.json", "r"))

    device = torch.device("cuda:0")
    config["network"]["device"] = device
    model = BioMedGPTV(config["network"])
    ckpt = torch.load("./ckpts/fusion_ckpts/biomedgpt_10b.pth")
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    prompt_sys = "You are working as an excellent assistant in biology. " + \
                 "Below a human gives the representation of a molecule or a protein. Answer some questions about it. "
    chat = Conversation(
        model=model, 
        processor_config=config["data"], 
        device=device,
        system=prompt_sys,
        roles=("Human", "Assistant"),
        sep="###",
        max_length=2048
    )
    chat.append_protein("MAKEDTLEFPGVVKELLPNATFRVELDNGHELIAVMAGKMRKNRIRVLAGDKVQVEMTPYDLSKGRINYRFK")
    questions = ["What are the official names of this protein?", "What is the function of this protein?"]
    for q in questions:
        print("Human: ", q)
        chat.ask(q)
        print("Assistant: ", chat.answer()[0])
    print("Chat reset.")
    chat.reset()
    chat.append_molecule("C[C@]12CCC(=O)C=C1CC[C@@H]3[C@@H]2C(=O)C[C@]\\\\4([C@H]3CC/C4=C/C(=O)OC)C")
    questions = ["Please describe this drug."]
    for q in questions:
        print("Human: ", q)
        chat.ask(q)
        print("Assistant: ", chat.answer()[0])