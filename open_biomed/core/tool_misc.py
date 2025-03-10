from typing import Union, Tuple, List

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from open_biomed.core.tool import Tool
from open_biomed.data import Molecule, Protein, Pocket
from open_biomed.core.pipeline import InferencePipeline

class ImportPocket(Tool):
    def __init__(self) -> None:
        super().__init__()

    def print_usage(self) -> str:
        return "Construct an OpenBioMed pocket object from several amino acids within a protein"
    
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
        return "Convert an OpenBioMed molecule object to a sdf file"

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
        return "Convert an OpenBioMed protein object to a pdb file"

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
                        model_ckpt="/AIRvePFS/dair/users/ailin/.cache/huggingface/hub/esmfold_v1/pytorch_model.bin",
                        device="cuda:1"
                    )
                prot = pipeline.run(protein=prot)[0][0]
            files.append(prot.save_pdb())
        return files


class MutationToSequence(Tool):
    def __init__(self) -> None:
        super().__init__()

    def print_usage(self) -> str:
        return 'Apply a single-site mutation to the wild-type sequence\n' + \
               'Inputs: {"protein": wild-type protein, "mutation": a mutation}\n' + \
               'Outputs: a mutated protein\n'

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
    """
    tool = ExportMolecule()
    molecule = Molecule.from_smiles("C1=CC=C(C=C1)C=O")
    print(tool.run(molecule)[0])
