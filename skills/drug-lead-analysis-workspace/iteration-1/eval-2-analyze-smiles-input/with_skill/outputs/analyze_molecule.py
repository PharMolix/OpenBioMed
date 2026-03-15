#!/usr/bin/env python
import sys
import os
sys.path.insert(0, '/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_dev')
os.chdir('/AIRvePFS/dair/luoyz-data/projects/OpenBioMed/OpenBioMed_dev')

from open_biomed.data import Molecule
from open_biomed.tools import TOOLS
from rdkit.Chem import Descriptors, Lipinski

# Step 1: Create the molecule from SMILES
smiles = 'CC(C)Cc1ccc(cc1)C(C)C(=O)O'
print(f'Creating molecule from SMILES: {smiles}')
molecule = Molecule.from_smiles(smiles)
molecule._add_name()
print(f'Molecule created successfully')

# Step 2: Calculate drug-likeness scores
print('\n=== Calculating Drug-likeness Scores ===')

# QED
qed_tool = TOOLS['molecule_qed']
qed_result, qed_msg = qed_tool.run(molecule=molecule)
print(f'QED: {qed_result[0]}')

# SA Score
sa_tool = TOOLS['molecule_sa']
sa_result, sa_msg = sa_tool.run(molecule=molecule)
print(f'SA Score: {sa_result[0]}')

# LogP
logp_tool = TOOLS['molecule_logp']
logp_result, logp_msg = logp_tool.run(molecule=molecule)
print(f'LogP: {logp_result[0]}')

# Lipinski
lipinski_tool = TOOLS['molecule_lipinski']
lipinski_result, lipinski_msg = lipinski_tool.run(molecule=molecule)
print(f'Lipinski rules satisfied: {lipinski_result[0]}')

# Additional properties
print('\n=== Additional Properties ===')
molecule._add_rdmol()
mol = molecule.rdmol
mw = Descriptors.ExactMolWt(mol)
hbd = Lipinski.NumHDonors(mol)
hba = Lipinski.NumHAcceptors(mol)
rotatable = Descriptors.NumRotatableBonds(mol)
num_atoms = mol.GetNumAtoms()
print(f'Molecular Weight: {mw:.2f} Da')
print(f'H-bond donors: {hbd}')
print(f'H-bond acceptors: {hba}')
print(f'Rotatable bonds: {rotatable}')
print(f'Number of atoms: {num_atoms}')

# Count Lipinski violations
violations = 0
if mw > 500: violations += 1
if logp_result[0] > 5: violations += 1
if hbd > 5: violations += 1
if hba > 10: violations += 1
print(f'Lipinski violations: {violations}')

print('\nAnalysis complete!')
print(f'\n--- SUMMARY ---')
print(f'SMILES: {smiles}')
print(f'QED Score: {qed_result[0]:.4f}')
print(f'SA Score: {sa_result[0]:.2f}')
print(f'LogP: {logp_result[0]:.2f}')
print(f'Lipinski violations: {violations}')
print(f'Molecular Weight: {mw:.2f} Da')
print(f'HBD: {hbd}, HBA: {hba}')
