from typing import Dict, List, Tuple

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging
import numpy as np
from rdkit import Chem
import re
import torch

from open_biomed.core.llm_provider import get_llm
from open_biomed.data import Molecule, Pocket, Text, calc_mol_fragment
from open_biomed.models.task_models.text_based_molecule_editing import TextBasedMoleculeEditingModel
from open_biomed.models.task_models.structure_based_drug_design import StructureTextBasedMoleculeOptimizationModel
from open_biomed.tools.tool_misc import ComplexInteractionAnalysis
from open_biomed.utils.config import Config
from open_biomed.utils.collator import ListCollator
from open_biomed.utils.featurizer import IdenticalFeaturizer, Featurized


class TextBasedMoleculeEditingAgent(TextBasedMoleculeEditingModel, StructureTextBasedMoleculeOptimizationModel):
    def __init__(self, model_cfg: Config) -> None:
        super().__init__(model_cfg)
        self.llm = get_llm(model_cfg.llm, temperature=model_cfg.temperature, stop_sequences=["</SMILES>"])
        self.system_prompt = """
You are a professional and helpful biomedical assistant who major in medicinal chemistry. Your task is to design modifications to a molecule to achieve a desired property described in text.

[Design Instructions]
1. Do not modify the molecule too much. Maintain the key features of the molecule if not specified in the text (e.g., shape, size, and functionality).
2. The modified molecule should be stable and easy to synthesize. New fragments should be common and stable.
3. Avoid introducing toxic groups such as aniline, hydrazine/azo, or Michael acceptors.
4. Show your thinking and design rationales. The rationales should be concise and informative.
5. Provide the SMILES string of the optimized molecule in the final answer with <SMILES>...</SMILES> tags.
"""
        if getattr(model_cfg, "scaffold_hopping", False):
            self.system_prompt += """
6. Your job is scaffold hopping. You should change the scaffold of the molecule and maintain the remaining key pharmacophores.
"""
        else:
            self.system_prompt += """
6. Your job is side chain modification. You should change the type or position of substituent groups on the backbone and maintain the overall structure.
"""

        self.tolerance = getattr(model_cfg, "tolerance", 3)

        self.featurizers = {
            "molecule": IdenticalFeaturizer(["molecule"]),
            "pocket": IdenticalFeaturizer(["pocket"]),
            "text": IdenticalFeaturizer(["text"]),
        }
        self.collators = {
            "molecule": ListCollator(),
            "pocket": ListCollator(),
            "text": ListCollator(),
        }
        for parent in reversed(type(self).__mro__[1:-1]):
            if hasattr(parent, '_add_task'):
                parent._add_task(self)

    def _analyze_mol_fragments(self, molecule: Molecule) -> str:
        atom2frag = np.array(calc_mol_fragment(molecule))
        frag_smiles = []
        for i in range(np.max(atom2frag) + 1):
            frag_idx = np.where(atom2frag == i)[0]
            frag_smiles.append(Chem.MolFragmentToSmiles(molecule.rdmol, frag_idx.tolist(), kekuleSmiles=True))
        frag_smiles.append(Chem.MolFragmentToSmiles(molecule.rdmol, list(range(molecule.get_num_atoms())), kekuleSmiles=True))
        molecule._add_conformer(mode="3D")

        ret = """
| Atom Type | X coordinate | Y coordinate | Z coordinate | Fragment SMILES |
|-----------|--------------|--------------|--------------|-----------------|
"""
        for i in range(molecule.get_num_atoms()):
            ret += f"| {molecule.rdmol.GetAtomWithIdx(i).GetSymbol()} | {molecule.conformer[i][0]:.4f} | {molecule.conformer[i][1]:.4f} | {molecule.conformer[i][2]:.4f} | {frag_smiles[atom2frag[i]]} |\n"
        return ret

    def _optimize_molecule(self, user_prompt: str) -> Tuple[Molecule, str]:
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt)
        ]
        print(user_prompt)
        # print(self.llm.invoke([HumanMessage(content="calculate 2+2")]).content)
        # print(self.llm.invoke([HumanMessage(content="Analyze the molecule Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1 and write a report")]).content)
        for i in range(self.tolerance):
            response = self.llm.invoke(messages).content
            if "<SMILES>" in response and not "</SMILES>" in response:
                response += "</SMILES>"
            print(response)
            smiles = re.search(r"<SMILES>(.*?)</SMILES>", response)
            fail_msg = None
            if smiles:
                try:
                    molecule = Molecule.from_smiles(smiles.group(1))
                    return molecule, None
                except Exception as e:
                    fail_msg = "Failed to construct molecule from SMILES string."
            else:
                fail_msg = "You should contain the SMILES string of the optimized molecule in the final answer with <SMILES>...</SMILES> tags, but there is none in your response. Please try again."
            if fail_msg:
                messages.append(AIMessage(content=response))
                messages.append(HumanMessage(content=fail_msg))
        return None, "Failed to optimize molecule after {} attempts.".format(self.tolerance)

    def forward_text_based_molecule_editing(self, 
        molecule: Featurized[Molecule], 
        text: Featurized[Text],
        label: Featurized[Molecule],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def predict_text_based_molecule_editing(self, 
        molecule: Featurized[Molecule], 
        text: Featurized[Text],
    ) -> List[Molecule]:
        optimized_molecules = []
        for mol, txt in zip(molecule, text):
            user_prompt = f"""
The SMILES string of the molecule is: 
<SMILES>{mol.smiles}</SMILES> 

{self._analyze_mol_fragments(mol)}

The desired property is: 
{txt.text} 

Please design modifications to the molecule to achieve the desired property. 
"""
            optimized_mol, fail_msg = self._optimize_molecule(user_prompt)
            if optimized_mol:
                optimized_molecules.append(optimized_mol)
            else:
                logging.warning(fail_msg)
                optimized_molecules.append(None)
        return optimized_molecules

    def forward_structure_text_based_molecule_optimization(self, 
        molecule: Featurized[Molecule], 
        pocket: Featurized[Pocket],
        text: Featurized[Text],
        label: Featurized[Molecule],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def predict_structure_text_based_molecule_optimization(self, 
        molecule: Featurized[Molecule], 
        pocket: Featurized[Pocket],
        text: Featurized[Text],
    ) -> List[Molecule]:
        optimized_molecules = []
        for mol, pck, txt in zip(molecule, pocket, text):
            file = pck.save_pdb()
            pck_pdb_lines = "".join(open(file, "r").readlines())

            user_prompt = f"""
The SMILES string of the molecule is: 
{mol.smiles}

The desired property is: 
{txt.str} 

Please design modifications to the molecule to achieve the desired property and maintain or improve the binding affinity to the pocket. 

Here is the fragment analysis of the molecule:
{self._analyze_mol_fragments(mol)}

Here are the atoms in the pocket in the PDB format:
{pck_pdb_lines}

The interaction between the molecule and the pocket is:
{ComplexInteractionAnalysis().run(mol, pck.orig_protein)[0][0]}
"""
            optimized_mol, fail_msg = self._optimize_molecule(user_prompt)
            if optimized_mol:
                from open_biomed.tasks.aidd_tasks.protein_molecule_docking import VinaDockTask
                task = VinaDockTask()
                score, optimized_mol = task.run(molecule=optimized_mol, protein=pck, mode="dock")[0][0]
                optimized_molecules.append(optimized_mol)
            else:
                logging.warning(fail_msg)
                optimized_molecules.append(None)
        return optimized_molecules