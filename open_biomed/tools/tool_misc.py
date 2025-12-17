from langchain_core.messages import SystemMessage, HumanMessage
from typing import Union, Tuple, List, Optional

import logging
import os
import sys
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(work_dir)

from open_biomed.core.llm_provider import get_llm
from open_biomed.tools.base_tool import Tool
from open_biomed.tools.web_request_tools import DBRequester
from open_biomed.data import Molecule, Protein, Pocket
from open_biomed.core.pipeline import InferencePipeline

class ImportPocket(Tool):
    def __init__(self) -> None:
        super().__init__()

    def print_usage(self) -> str:
        return """
Usage: Construct a pocket object from several amino acids within a protein
Inputs: {"protein": Protein (an OpenBioMed Protein object), "indices": List[int] (a list of indices of the amino acids within the protein)}
Outputs: Pocket (an OpenBioMed Pocket object)
"""
    
    def run(self, protein: Union[Protein, List[Protein]], indices: Union[List[int], List[List[int]]]) -> Tuple[List[Pocket], List[str]]:
        if isinstance(protein, Protein):
            protein = [protein]
            if isinstance(indices, str):
                indices = [int(i) - 1 for i in indices.split(";")]
            indices = [indices]
        pockets, files = [], []
        for i in range(len(protein)):
            pocket = Pocket.from_protein_subseq(protein[i], indices[i])
            pockets.append(pocket)
            files.append(pocket.save_binary())
        return pockets, files

class ExportMolecule(Tool):
    def __init__(self) -> None:
        super().__init__()

    def print_usage(self) -> str:
        return """
Usage: Convert an OpenBioMed molecule object to a sdf file
Inputs: {"molecule": Molecule (an OpenBioMed Molecule object)}
Outputs: str (the path to the sdf file)
"""

    def run(self, molecule: Union[Molecule, List[Molecule]]) -> Tuple[List[str], List[str]]:
        if isinstance(molecule, Molecule):
            molecule = [molecule]
        files = []
        for mol in molecule:
            files.append(mol.save_sdf())
        return files

class ExportProtein(Tool):
    def __init__(self) -> None:
        super().__init__()

    def print_usage(self) -> str:
        return """
Usage: Convert an OpenBioMed protein object to a pdb file
Inputs: {"protein": Protein (an OpenBioMed Protein object)}
Outputs: str (the path to the pdb file)
"""

    def run(self, protein: Union[Protein, List[Protein]]) -> Tuple[List[str], List[str]]:
        if isinstance(protein, Protein):
            protein = [protein]
        files = []
        pipeline = None
        for prot in protein:
            if getattr(prot, "conformer", None) is None:
                if pipeline is None:
                    pipeline = InferencePipeline(
                        task="protein_folding",
                        model="esmfold",
                        model_ckpt="./checkpoints/server/esmfold.ckpt",
                        device="cuda:1"
                    )
                prot = pipeline.run(protein=prot)[0][0]
            files.append(prot.save_pdb())
        return files


class MutationToSequence(Tool):
    def __init__(self) -> None:
        super().__init__()

    def print_usage(self) -> str:
        return """
Usage: Apply a single-site mutation to the wild-type sequence
Inputs: {"protein": Protein (a wild-type protein), "mutation": str (a mutation, e.g. "M123A")}
Outputs: Protein (a mutated protein object)
"""

    def run(self, protein: Union[List[Protein], Protein], mutation: Union[List[str], str]) -> Tuple[List[Protein], List[str]]:
        if not isinstance(protein, list):
            protein = [protein]
            mutation = [mutation]
        mutants, files = [], []
        for i in range(len(protein)):
            seq = protein[i].sequence
            pos = int(mutation[i][1:-1])
            mutant = Protein.from_fasta(seq[:pos - 1] + mutation[i][-1] + seq[pos:])
            mutant.name = protein[i].name + "_" + mutation[i]
            mutants.append(mutant)
            files.append(mutant.save_binary())
        return mutants, files

class LLMSummarize(Tool):
    def __init__(self, llm: str="deepseek-chat", system_prompt: Optional[str]=None) -> None:
        super().__init__()
        self.llm = get_llm(llm, stop_sequences=["</summary>"])
        if system_prompt is None:
            self.system_prompt = "You are a biological data analyst. Given the raw text of the metadata, you should summarize the metadata in a concise and informative way. Do not miss any important information in the metadata. Wrap the summary in <summary>...</summary> tags."
        else:
            self.system_prompt = system_prompt

    def print_usage(self) -> str:
        return """
Usage: Summarize a given Document
Inputs: {"content": str (a Python String of the document to summarize)}
Outputs: str (a concise and informative summary of the document)
"""

    def run(self, content: str) -> str:
        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=content)
        ]).content
        while len(response) < 1 or "<summary>" not in response:
            logging.info("Receiving invalid response from LLM, retrying...")
            response = self.llm.invoke([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=content)
            ]).content
        return [response.lstrip("<summary>").rstrip("</summary>")], ["Summary completed, the summary is: " + response.lstrip("<summary>").rstrip("</summary>")]

class ExtractAllMoleculesFromPDB(Tool):
    def __init__(self) -> None:
        super().__init__()

    def print_usage(self) -> str:
        return """
Usage: Extract proteins, ligands, and ions from a PDB file
Inputs: {"pdb_file": str (the path to the PDB file)}
Outputs: List[Tuple[str, str, Molecule | Protein]] (a list of tuples, the first element is the type of the molecule, which can be "molecule", "ion", or "protein", the second element is the chain id, the third element is the molecule or protein object)
"""

    def run(self, pdb_file: str) -> List[Tuple[str, Union[Molecule, Protein]]]:
        results = []
        output_metadata = ""
        num_ions = 0
        num_molecules = 0
        pdb_id = pdb_file.split("/")[-1].lstrip("pdb_").rstrip(".pdb")

        def is_metal(element):
            metals = {
                'LI', 'NA', 'K', 'RB', 'CS', 'FR',       # Alkali metals
                'BE', 'MG', 'CA', 'SR', 'BA', 'RA',      # Alkaline earth metals
                'SC', 'TI', 'V', 'CR', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN',
                'Y', 'ZR', 'NB', 'MO', 'TC', 'RU', 'RH', 'PD', 'AG', 'CD',
                'HF', 'TA', 'W', 'RE', 'OS', 'IR', 'PT', 'AU', 'HG',
                'AL', 'GA', 'IN', 'TL', 'SN', 'PB', 'BI', 'PO', 'LU', 'LA',
                'CE', 'PR', 'ND', 'PM', 'SM', 'EU', 'GD', 'TB', 'DY', 'HO',
                'ER', 'TM', 'YB', 'TH', 'PA', 'U', 'NP', 'PU', 'AM', 'CM',
                'BK', 'CF', 'ES', 'FM', 'MD', 'NO', 'LR', 'AC'
            }
            return element and element.strip().upper() in metals

        # Parse the pdb_file manually, grouping ATOM/HETATM lines by chain and residue
        chains = {}
        hetero_residues = {}  # key: (res_name, chain_id, res_seq)
        hetatm_lines = []
        conect_lines = []
        with open(pdb_file, "r") as f:
            for line in f:
                record = line[0:6].strip()
                if record == "ATOM":
                    chain_id = line[21].strip()
                    if chain_id == "":
                        chain_id = "_"
                    chains.setdefault(chain_id, []).append(line)
                elif record == "HETATM":
                    res_name = line[17:20].strip()
                    chain_id = line[21].strip()
                    if chain_id == "":
                        chain_id = "_"
                    auth_seq_id = line[22:26].strip()
                    hetatm_lines.append(line)
                    key = (res_name, chain_id, auth_seq_id)
                    hetero_residues.setdefault(key, []).append(line)
                elif record == "CONECT":
                    conect_lines.append(line)

        # Extract proteins, keeping each chain separate
        for chain_id, atom_lines in chains.items():
            # Write chain structure to a temporary string
            protein = Protein.from_pdb(atom_lines)
            protein.name = f"{pdb_id}_{chain_id}"
            results.append(("protein", chain_id, protein))
            output_metadata += f"protein chain {chain_id}: {protein.sequence}\n"

        # Extract non-polymer molecules, grouping HETATM residues
        for (res_name, chain_id, auth_seq_id), lines_in_res in hetero_residues.items():
            # Exclude waters
            if res_name == 'HOH':
                continue
            # Try to determine if this is a metal ion (single atom, element is a metal)
            is_single_atom = len(lines_in_res) == 1
            element = lines_in_res[0][76:78].strip() if is_single_atom and len(lines_in_res[0]) >= 78 else ""
            if is_single_atom:
                if not is_metal(res_name) and not is_metal(element):
                    continue
                else:
                    atm_type = "ion"
                    num_ions += 1
            else:
                atm_type = "molecule"
                num_molecules += 1
            # Query the molecule since pdb block do not specify bond types
            requester = DBRequester(timeout=30)
            requester.db_url = f"https://models.rcsb.org/v1/{pdb_id}/" + "ligand?auth_seq_id={accession}&encoding=sdf" 
            try:
                filename = f"{pdb_id}_{res_name}_{chain_id}_{auth_seq_id}"
                content = requester.run(auth_seq_id)[0][0]
                with open(f"{work_dir}/tmp/{filename}.sdf", "w") as f:
                    f.write(content)
                molecule = Molecule.from_sdf_file(f"{work_dir}/tmp/{filename}.sdf")
                results.append((atm_type, chain_id, molecule))
            except Exception as e:
                print(e)
                molecule = Molecule.from_pdb(lines_in_res)
                continue
            output_metadata += f"{atm_type}: {molecule.smiles}\n"

        output_metadata += f"Total {len(chains)} protein chains, {num_molecules} molecules and {num_ions} ions extracted from {pdb_file}"
        return [results], [output_metadata]

if __name__ == "__main__":
    """
    tool = ExportProtein()
    protein = Protein.from_fasta("MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQRVEDAFYTLVREIRQYRLKKISKEEKTPGCVKIKKCIIM")
    tool.run(protein)
    tool = ImportPocket()
    protein = Protein.from_pdb_file("./tmp/sbdd/4xli_B.pdb")
    molecule = Molecule.from_sdf_file("./tmp/sbdd/4xli_B_ref.sdf")
    pocket = Pocket.from_protein_ref_ligand(protein, molecule)
    print(pocket)
    print(tool.run(protein, pocket.orig_indices)[0])
    tool = ExportMolecule()
    molecule = Molecule.from_smiles("C1=CC=C(C=C1)C=O")
    print(tool.run(molecule)[0])
    """
    tool = ExtractAllMoleculesFromPDB()
    results = tool.run("./tmp/4xli.pdb")
    from open_biomed.data.molecule import check_identical_molecules, molecule_fingerprint_similarity
    mol = Molecule.from_smiles("CC1=C(C(=CC=C1)Cl)NC(=O)C2=CN=C(S2)NC3=CC(=NC(=N3)C)N4CCN(CC4)CCO")
    print(check_identical_molecules(mol, results[0][0][2][2]))
    print(molecule_fingerprint_similarity(mol, results[0][0][2][2]))
    print(results[0][0][2][2].rdmol.GetRingInfo().AtomRings())
    print(results)
