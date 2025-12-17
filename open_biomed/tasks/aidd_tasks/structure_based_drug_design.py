from typing import Optional, Any, Dict, List

import logging
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

from open_biomed.data import Molecule, Protein
from open_biomed.tasks.base_task import BaseTask, DefaultDataModule, DefaultModelWrapper
from open_biomed.utils.collator import Collator
from open_biomed.utils.config import Config, Struct
from open_biomed.utils.featurizer import Featurizer

class StructureBasedDrugDesign(BaseTask):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def print_usage() -> str:
        return "\n".join([
            'Structure-based drug design.',
            'Inputs: {"pocket": Pocket (an OpenBioMed Pocket object)} REMARK: we recommend using Pocket.from_protein_ref_ligand(protein, ligand) to construct the pocket object so that the generation is more likely to succeed. The ligand should be a reference molecule that is already bound to the protein and should NOT be constructed from scratch.',
            "Outputs: Molecule (an OpenBioMed Molecule object that is likely to bind with the pocket)"
        ])

    @staticmethod
    def get_datamodule(dataset_cfg: Config, featurizer: Featurizer, collator: Collator) -> pl.LightningDataModule:
        return DefaultDataModule("structure_based_drug_design", dataset_cfg, featurizer, collator)

    @staticmethod
    def get_model_wrapper(model_cfg: Config, train_cfg: Config) -> pl.LightningModule:
        return DefaultModelWrapper("structure_based_drug_design", model_cfg, train_cfg)

    @staticmethod
    def get_callbacks(callback_cfg: Optional[Config]=None) -> pl.Callback:
        return StructureBasedDrugDesignEvaluationCallback()

    @staticmethod
    def get_monitor_cfg() -> Struct:
        return Struct(
            name="val/vina_score/median",
            output_str="-{val_vina_score_med:.4f}",
            mode="min",
        )
    
def calc_vina_molecule_metrics(molecule: Molecule, protein: Protein, calculate_vina_dock: bool=True) -> Dict[str, float]:
    metrics = {}

    # Vina scores
    from open_biomed.tasks.aidd_tasks.protein_molecule_docking import VinaDockTask
    modes = ["min", "dock", "score"] if calculate_vina_dock else ["min", "score"]
    for mode in modes:
        vina_task = VinaDockTask(mode=mode)
        metrics[f"vina_{mode}"] = vina_task.run(molecule, protein)[0][0]

    # Molecule property metrics
    molecule._add_smiles()
    metrics["completeness"] = 0 if "." in molecule.smiles else 1
    metrics["qed"] = molecule.calc_qed()
    metrics["sa"] = molecule.calc_sa()
    metrics["logp"] = molecule.calc_logp()
    metrics["lipinski"] = molecule.calc_lipinski()

    # Success rate
    if calculate_vina_dock:
        metrics["success"] = 1 if metrics["vina_dock"] < -8.18 and metrics["qed"] > 0.25 and metrics["sa"] > 0.59 else 0
    else:
        metrics["success"] = 1 if metrics["vina_min"] < -8.18 and metrics["qed"] > 0.25 and metrics["sa"] > 0.59 else 0
    return metrics

BOND_TYPE_TO_ID = {
    Chem.rdchem.BondType.UNSPECIFIED: 0,
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}
EVAL_BOND_TYPES = [
    (6, 6, 1), #	CC	716	29.6%	1.2857153558712793	1.696778883283098	0.004110635274118186
    (6, 6, 4), #	C:C	500	20.7%	1.2981754588686738	1.5429516779717267	0.002447762191030529
    (6, 8, 1), #	CO	336	13.9%	1.217717567891834	1.592581263775381	0.0037486369588354694
    (6, 7, 1), #	CN	245	10.1%	1.2412786652760066	1.609101379383609	0.0036782271410760246
    (6, 7, 4), #	C:N	213	8.8%	1.2781037555594505	1.4881754271876604	0.002100716716282098
]
EVAL_BOND_TYPES_STR = ["CC", "C:C", "CO", "CN", "C:N"]
EVAL_BOND_ANGLES = [
    (6, 1, 6, 1, 6), #	CCC	521	18.1%	59.52230720788234	135.50315793532704	0.759808507274447
    (6, 4, 6, 4, 6), #	C:C:C	460	16.0%	101.54806405949785	127.54928623790771	0.2600122217840986
    (6, 1, 6, 1, 8), #	CCO	274	9.5%	57.19735111082594	136.5409407542893	0.7934358964346336
]
EVAL_BOND_ANGLES_STR = ["CCC", "C:C:C", "CCO"]

def calc_bond_length_profile(molecule: Molecule) -> Dict[str, List[float]]:
    distance = molecule.calc_distance()
    bond_len_profile = {k: [] for k in EVAL_BOND_TYPES}
    for bond in molecule.rdmol.GetBonds():
        s_sym = bond.GetBeginAtom().GetAtomicNum()
        t_sym = bond.GetEndAtom().GetAtomicNum()
        s_idx, t_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = BOND_TYPE_TO_ID[bond.GetBondType()]
        if s_idx > t_idx:
            s_idx, t_idx = t_idx, s_idx
        if (s_sym, t_sym, bond_type) in EVAL_BOND_TYPES:
            bond_len_profile[(s_sym, t_sym, bond_type)].append(distance[s_idx, t_idx])
    return bond_len_profile

def calc_bond_angle_profile(molecule: Molecule) -> Dict[str, List[float]]:
    bond_angle_profile = {k: [] for k in EVAL_BOND_ANGLES}
    for bond1 in molecule.rdmol.GetBonds():
        atom1, atom2 = bond1.GetBeginAtom(), bond1.GetEndAtom()
        for bond2 in atom2.GetBonds():
            atom3 = bond2.GetOtherAtom(atom2)
            if atom3.GetIdx() == atom1.GetIdx():
                continue
            try:
                angle = rdMolTransforms.GetAngleDeg(molecule.rdmol.GetConformer(), atom1.GetIdx(), atom2.GetIdx(), atom3.GetIdx())
                atom1_sym = atom1.GetAtomicNum()
                atom2_sym = atom2.GetAtomicNum()
                atom3_sym = atom3.GetAtomicNum()
                bond1_type = BOND_TYPE_TO_ID[bond1.GetBondType()]
                bond2_type = BOND_TYPE_TO_ID[bond2.GetBondType()]
                if atom1_sym > atom3_sym:
                    atom1_sym, atom3_sym = atom3_sym, atom1_sym
                    bond1_type, bond2_type = bond2_type, bond1_type
                if (atom1_sym, bond1_type, atom2_sym, bond2_type, atom3_sym) in EVAL_BOND_ANGLES:
                    bond_angle_profile[(atom1_sym, bond1_type, atom2_sym, bond2_type, atom3_sym)].append(angle)
    
            except Exception as e:
                logging.error(f"Error calculating bond angle for {molecule.name}: {e}")
                continue
            
    return bond_angle_profile

class StructureBasedDrugDesignEvaluationCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = []
        self.eval_dataset = None

    def init_metrics(self) -> None:
        self.metrics = {"valid": []}
        self.bond_len_profile_ref = {}
        self.bond_len_profile_pred = {}
        self.bond_angle_profile_ref = {}
        self.bond_angle_profile_pred = {}
        for elem in EVAL_BOND_TYPES:
            self.bond_len_profile_ref[elem] = []
            self.bond_len_profile_pred[elem] = []
        for elem in EVAL_BOND_ANGLES:
            self.bond_angle_profile_ref[elem] = []
            self.bond_angle_profile_pred[elem] = []

    def on_validation_batch_end(self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: STEP_OUTPUT, 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.outputs.extend(outputs)
        if batch_idx == 0:
            for i in range(1):
                logging.info(f"Generated Molecule: {self.outputs[i]}")

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_validation_epoch_start(trainer, pl_module)
        self.status = "val"
        self.outputs = []
        self.eval_dataset = trainer.val_dataloaders.dataset

    def on_validation_epoch_end(self,
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ) -> None:
        for i in tqdm(range(len(self.outputs)), desc="Calculating SBDD metrics"):
            if self.outputs[i] is None:
                self.metrics["valid"].append(0)
                continue
            self.metrics["valid"].append(1)
            cur_metrics = calc_vina_molecule_metrics(self.outputs[i], self.eval_dataset.proteins[i])
            for k, v in cur_metrics.items():
                if not k in self.metrics:
                    self.metrics[k] = []
                self.metrics[k].append(v)
            
            # Calculate bond distance & angle profiles
            cur_len_profile_ref = calc_bond_length_profile(self.eval_dataset.molecules[i])
            for k, v in cur_len_profile_ref.items():
                self.bond_len_profile_ref[k].extend(v)
            cur_len_profile_pred = calc_bond_length_profile(self.outputs[i])
            for k, v in cur_len_profile_pred.items():
                self.bond_len_profile_pred[k].extend(v)
            
            cur_angle_profile_ref = calc_bond_angle_profile(self.eval_dataset.molecules[i])
            for k, v in cur_angle_profile_ref.items():
                self.bond_angle_profile_ref[k].extend(v)
            cur_angle_profile_pred = calc_bond_angle_profile(self.outputs[i])
            for k, v in cur_angle_profile_pred.items():
                self.bond_angle_profile_pred[k].extend(v)

            # Evaluate clash
            """
            try:
                from posecheck import PoseCheck
                pc = PoseCheck()
                pdb_file = self.eval_dataset.pockets[i].save_pdb()
                pc.load_protein_from_pdb(pdb_file)
                pc.load_ligands_from_mols([self.outputs[i].rdmol])
                clash = pc.calculate_clashes()[0]
                if "clash_num" not in self.metrics:
                    self.metrics["clash_num"] = []
                    self.metrics["atom_num"] = []
                self.metrics["clash_num"].append(clash)
                self.metrics["atom_num"].append(self.outputs[i].get_num_atoms())
            except ImportError:
                logging.error("posecheck is not installed. Please install it following README.md.")
            """
            
        # WARNING: the results are averaged over all test repeats, so only the last repeat should be used
        output_metrics = {}
        for vina_metrics in ["vina_min", "vina_dock", "vina_score"]:
            output_metrics[f"{self.status}/{vina_metrics}/median"] = np.median(self.metrics[vina_metrics])
            output_metrics[f"{self.status}/{vina_metrics}/mean"] = np.mean(self.metrics[vina_metrics])
        for metric in ["valid", "qed", "sa", "logp", "lipinski", "success"]:
            output_metrics[f"{self.status}/{metric}"] = np.mean(self.metrics[metric])
        # output_metrics[f"{self.status}/clash_ratio_atom"] = np.sum(self.metrics["clash_num"]) / np.sum(self.metrics["atom_num"])
        # output_metrics[f"{self.status}/clash_ratio_molecule"] = np.mean(np.array(self.metrics["clash_num"]) > 0)

        for i, bond_type in enumerate(EVAL_BOND_TYPES):
            bond_len_dist_ref = np.histogram(self.bond_len_profile_ref[bond_type], bins=np.arange(1.1, 1.7000001, 0.005), density=True)[0]
            bond_len_dist_pred = np.histogram(self.bond_len_profile_pred[bond_type], bins=np.arange(1.1, 1.7000001, 0.005), density=True)[0]
            jsd = jensenshannon(bond_len_dist_ref, bond_len_dist_pred)
            output_metrics[f"{self.status}/JSD_bond_len_{EVAL_BOND_TYPES_STR[i]}"] = 1 if np.isnan(jsd) else jsd
        
        for i, bond_angle in enumerate(EVAL_BOND_ANGLES):
            bond_angle_dist_ref = np.histogram(self.bond_angle_profile_ref[bond_angle], bins=np.arange(100, 140.01, 0.25), density=True)[0]
            bond_angle_dist_pred = np.histogram(self.bond_angle_profile_pred[bond_angle], bins=np.arange(100, 140.01, 0.25), density=True)[0]
            jsd = jensenshannon(bond_angle_dist_ref, bond_angle_dist_pred)
            output_metrics[f"{self.status}/JSD_bond_angle_{EVAL_BOND_ANGLES_STR[i]}"] = 1 if np.isnan(jsd) else jsd

        pl_module.log_dict(output_metrics)
    
    def on_test_batch_end(self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # For SBDD, we perform test multiple times
        super().on_test_epoch_start(trainer, pl_module)

        if getattr(self, "status", None) != "test":
            # First time test, initialize the stored results
            self.status = "test"
            self.init_metrics()
            self.eval_dataset = trainer.test_dataloaders.dataset
        self.outputs = []
        

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.on_validation_epoch_end(trainer, pl_module)
        
