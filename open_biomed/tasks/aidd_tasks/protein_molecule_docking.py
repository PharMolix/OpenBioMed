from typing import List, Optional, Union, Dict, Any, Tuple

import contextlib
import copy
import numpy as np
import os
from tqdm import tqdm
import sys
import pytorch_lightning as pl

from open_biomed.tools.base_tool import Tool
from open_biomed.data import Molecule, Protein, calc_mol_rmsd, mol_array_to_conformer
from open_biomed.tasks.base_task import BaseTask, DefaultDataModule, DefaultModelWrapper
from open_biomed.utils.collator import Collator
from open_biomed.utils.config import Config, Struct
from open_biomed.utils.featurizer import Featurizer

class PocketMoleculeDocking(BaseTask):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def print_usage() -> str:
        return "\n".join([
            'Ligand-pocket docking.',
            'Inputs: {"molecule": Molecule (an OpenBioMed Molecule object), "pocket": Pocket (an OpenBioMed Pocket object)}',
            "Outputs: Molecule (an OpenBioMed Molecule object with 3D coordinates indicating the binding pose, which could be accessed by .conformer as a numpy array)"
        ])

    @staticmethod
    def get_datamodule(dataset_cfg: Config, featurizer: Featurizer, collator: Collator) -> pl.LightningDataModule:
        return DefaultDataModule("pocket_molecule_docking", dataset_cfg, featurizer, collator)

    @staticmethod
    def get_model_wrapper(model_cfg: Config, train_cfg: Config) -> pl.LightningModule:
        return DefaultModelWrapper("pocket_molecule_docking", model_cfg, train_cfg)

    @staticmethod
    def get_callbacks(callback_cfg: Optional[Config]=None) -> pl.Callback:
        if callback_cfg is None or getattr(callback_cfg, "rmsd", False):
            return DockingRMSDEvaluationCallback()
        else:
            try:
                return DockingPoseBustersEvaluationCallback()
            except ImportError:
                return DockingRMSDEvaluationCallback()

    @staticmethod
    def get_monitor_cfg() -> Struct:
        return Struct(
            name="val/rmsd_mean",
            output_str="-{val_rmsd_mean:.4f}",
            mode="min",
        )

def pbcheck_single(mol_pred: List[Molecule], mol_gt: Molecule, protein: Protein, buster: object) -> List[float]:
    pdb_path = protein.save_pdb()
    for mol in mol_pred:
        mol._add_rdmol()
    buster_result = buster.bust([mol.rdmol for mol in mol_pred], mol_gt.rdmol, pdb_path)
    buster_result = buster_result.reset_index().iloc[0].to_dict()
    return buster_result

def aggregate_pb_results(pb_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    pbvalid_mol_keys = [
        "mol_pred_loaded",
        "mol_cond_loaded",
        "sanitization",
        "inchi_convertible",
        "all_atoms_connected",
        "bond_lengths",
        "bond_angles",
        "internal_steric_clash",
        "aromatic_ring_flatness",
        "double_bond_flatness",
        "internal_energy",
    ]

    pbvalid_dock_keys = [
        "protein-ligand_maximum_distance",
        "minimum_distance_to_protein",
        "minimum_distance_to_organic_cofactors",
        "minimum_distance_to_inorganic_cofactors",
        "minimum_distance_to_waters",
        "volume_overlap_with_protein",
        "volume_overlap_with_organic_cofactors",
        "volume_overlap_with_inorganic_cofactors",
        "volume_overlap_with_waters",
    ]

    pbvalid_mol = True
    for key in pbvalid_mol_keys:
        pbvalid_mol &= pb_results[0][key]
    pbvalid_dock = True
    for key in pbvalid_dock_keys:
        pbvalid_dock &= pb_results[0][key]
    return {
        "pbvalid_mol": pbvalid_mol,
        "pbvalid_dock": pbvalid_dock,
        "pbvalid": pbvalid_mol and pbvalid_dock,
    }

class DockingRMSDEvaluationCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = []
        self.rmsds = []
        self.eval_dataset = None

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Dict[str, Any], batch: Dict[str, Any], batch_idx: int, dataloader_idx: int=0) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.outputs.extend(outputs)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_validation_epoch_start(trainer, pl_module)
        self.status = "val"
        self.outputs = []
        self.eval_dataset = trainer.val_dataloaders.dataset

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        all_rmsd = []
        for i in tqdm(range(len(self.outputs)), desc="Calculating docking RMSD"):
            # The decoded molecule may be different due to post-processing such as kekulization, so we only update the conformer.
            self.outputs[i].rdmol = copy.deepcopy(self.eval_dataset.molecules[i].rdmol)
            self.outputs[i].rdmol.RemoveAllConformers()
            self.outputs[i].rdmol.AddConformer(mol_array_to_conformer(self.outputs[i].conformer))
            rmsd = calc_mol_rmsd(self.outputs[i], self.eval_dataset.molecules[i])
            all_rmsd.append(rmsd)
        self.rmsds.append(all_rmsd)
        pl_module.log(f"{self.status}/rmsd_mean", np.mean(all_rmsd))
        pl_module.log(f"{self.status}/rmsd_median", np.median(all_rmsd))
        pl_module.log(f"{self.status}/rmsd < 2Å (%)", np.mean([rmsd < 2 for rmsd in all_rmsd]))

    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Dict[str, Any], batch: Dict[str, Any], batch_idx: int, dataloader_idx: int=0) -> None:
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # For docking, we may generate multiple poses.
        super().on_test_epoch_start(trainer, pl_module)
        if getattr(self, "status", None) != "test":
            self.status = "test"
            self.rmsds = []
            self.eval_dataset = trainer.test_dataloaders.dataset
        self.outputs = []

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.on_validation_epoch_end(trainer, pl_module)

class DockingPoseBustersEvaluationCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = []
        self.rmsds = []
        self.pbresults = {}
        self.eval_dataset = None
        try:
            from posebusters import PoseBusters
            self.buster = PoseBusters()
        except ImportError:
            raise ImportError("PoseBusters not installed. Please install it using `pip install posebusters==0.3.1`.")

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Dict[str, Any], batch: Dict[str, Any], batch_idx: int, dataloader_idx: int=0) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.outputs.extend(outputs)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_validation_epoch_start(trainer, pl_module)
        self.status = "val"
        self.outputs = []
        self.eval_dataset = trainer.val_dataloaders.dataset

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        all_rmsd = []
        all_pbresults = {}
        for i in tqdm(range(len(self.outputs)), desc="Calculating docking RMSD"):
            rmsd = calc_mol_rmsd(self.outputs[i], self.eval_dataset.molecules[i])
            all_rmsd.append(rmsd)
            pb_results = pbcheck_single(self.outputs[i], self.eval_dataset.molecules[i], self.eval_dataset.proteins[i], self.buster)
            for key, value in pb_results.items():
                if key not in all_pbresults:
                    all_pbresults[key] = []
                all_pbresults[key].append(value)
            agg = aggregate_pb_results(pb_results)
            for key, value in agg.items():
                if key not in all_pbresults:
                    all_pbresults[key] = []
                all_pbresults[key].append(value)
            if "pbvalid & rmsd < 2Å" not in all_pbresults:
                all_pbresults["pbvalid & rmsd < 2Å"] = []
            all_pbresults["pbvalid & rmsd < 2Å"].append(rmsd < 2 and agg["pbvalid"])
        self.rmsds.append(all_rmsd)
        for key, value in all_pbresults.items():
            if key not in self.pbresults:
                self.pbresults[key] = []
            self.pbresults[key].append(value)
            pl_module.log(f"{self.status}/{key}", np.mean(value))
        pl_module.log(f"{self.status}/rmsd_mean", np.mean(all_rmsd))
        pl_module.log(f"{self.status}/rmsd_median", np.median(all_rmsd))

    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Dict[str, Any], batch: Dict[str, Any], batch_idx: int, dataloader_idx: int=0) -> None:
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # For docking, we may generate multiple poses.
        super().on_test_epoch_start(trainer, pl_module)
        if getattr(self, "status", None) != "test":
            self.status = "test"
            self.rmsds = []
            self.pbresults = {}
            self.eval_dataset = trainer.test_dataloaders.dataset
        self.outputs = []

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.on_validation_epoch_end(trainer, pl_module)

class VinaDockTask(Tool):
    def __init__(self, mode: str="dock") -> None:
        self.mode = mode
        
        python_path = sys.executable
        conda_env_root = os.path.dirname(os.path.dirname(python_path))
        self.pdb2pqr_path = os.path.join(conda_env_root, 'bin', 'pdb2pqr30')

    def print_usage(self) -> str:
        return "\n".join([
            'Ligand-receptor docking.',
            'Inputs: {"molecule": Molecule (an OpenBioMed Molecule object), "protein": Protein (an OpenBioMed Protein object)}',
            "Outputs: A float number indicating the AutoDockVina score of the binding."
        ])

    def run(self, molecule: Molecule, protein: Protein) -> Tuple[List[float], List[str]]:
        sdf_file = molecule.save_sdf()
        pdb_file = protein.save_pdb()
        pos = np.array(molecule.conformer)
        center = (pos.max(0) + pos.min(0)) / 2
        size = pos.max(0) - pos.min(0) + 5
        try:
            from openbabel import pybel
            from meeko import MoleculePreparation
            import subprocess
            from vina import Vina
            import AutoDockTools

            ob_mol = next(pybel.readfile("sdf", sdf_file))
            lig_pdbqt = sdf_file.replace(".sdf", ".pdbqt")
            if not os.path.exists(lig_pdbqt):
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        preparator = MoleculePreparation()
                        preparator.prepare(ob_mol.OBMol)
                        preparator.write_pdbqt_file(lig_pdbqt)
            
            prot_pqr = pdb_file.replace(".pdb", ".pqr")
            if not os.path.exists(prot_pqr):
                subprocess.Popen([self.pdb2pqr_path,'--ff=AMBER', pdb_file, prot_pqr],
                            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()
            prot_pdbqt = pdb_file.replace(".pdb", ".pdbqt")
            if not os.path.exists(prot_pdbqt):
                prepare_receptor = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')
                subprocess.Popen(['python3', prepare_receptor, '-r', prot_pqr, '-o', prot_pdbqt],
                                stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()
            
            v = Vina(sf_name='vina', seed=0, verbosity=0)
            v.set_receptor(prot_pdbqt)
            v.set_ligand_from_file(lig_pdbqt)
            v.compute_vina_maps(center=center, box_size=size)
            if self.mode == "min":
                score = v.optimize()[0]
                pose_file = f"./tmp/{molecule.name}_{protein.name}_pose"
                with open(pose_file, "w") as f:
                    v.write_pose(pose_file, overwrite=True)
            elif self.mode == 'dock':
                v.dock(exhaustiveness=8, n_poses=1)
                score = v.energies(n_poses=1)[0][0]
                pose_file = "None"
            elif self.mode == 'score':
                score = v.score()[0]
                pose_file = "None"
            return [score], [pose_file]
        except ImportError:
            print("AutoDockVina not installed. This function return 0.0.")
            return [0.0], ["0.0"]

class ComplexInteractionAnalysis(Tool):
    def __init__(self) -> None:
        super().__init__()

    def print_usage(self) -> str:
        return "\n".join([
            'Analyze the interactions between a ligand and a protein.',
            'Inputs: {"molecule": Molecule (an OpenBioMed Molecule object), "protein": Protein (an OpenBioMed Protein object)}',
            "Outputs: str (a textual report of the interactions between the ligand and the protein)"
        ])

    def run(self, molecule: Union[Molecule, List[Molecule]], protein: Union[Protein, List[Protein]]) -> Tuple[List[str], List[str]]:
        try:
            from rdkit import Chem
            from plip.structure.preparation import PDBComplex
            from plip.exchange.report import BindingSiteReport

            if isinstance(molecule, Molecule):
                molecule = [molecule]
            if isinstance(protein, Protein):
                protein = [protein]
            all_report = []
            for mol, prot in zip(molecule, protein):
                pdb_file = prot.save_pdb()
                mol._add_name()
                rdmol = mol.rdmol
                rdprotein = Chem.MolFromPDBFile(pdb_file)
                rdcomplex = Chem.CombineMols(rdmol, rdprotein)
                complex_pdb_file = pdb_file.replace(".pdb", f"_{mol.name}_complex.pdb")
                Chem.MolToPDBFile(rdcomplex, complex_pdb_file)
                complex = PDBComplex()
                complex.read_pdb_file(complex_pdb_file)
                complex.analyze()
                all_report.append("")
                for key in complex.interaction_sets:
                    if key.startswith("UNL"):
                        interactions = complex.interaction_sets[key]
                        report = BindingSiteReport(interactions)
                        report_txt = ";".join(report.generate_txt())
                        all_report[-1] += f"{key}:\n{report_txt}\n"
            return all_report, all_report
        except ImportError:
            raise ImportError("PLIP not installed. Please install it using `pip install plip`.")