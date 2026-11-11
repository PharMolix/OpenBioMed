import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from open_biomed.data.molecule import Molecule

@pytest.mark.parametrize('smiles', ['CCO', 'c1ccccc1', 'CC(=O)OC1=CC=CC=C1C(=O)O'])
def test_no_conformer_retains_molecular_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol.GetNumConformers() == 0
    result = Molecule.from_rdmol(mol)
    assert result.rdmol is mol
    assert result.smiles == Chem.MolToSmiles(mol)
    assert result.conformer is None
    assert mol.GetNumConformers() == 0

@pytest.mark.parametrize('conformer_id', [0, 7])
def test_existing_conformer_is_preserved(conformer_id):
    mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    mol.GetConformer().SetId(conformer_id)
    expected = np.array(mol.GetConformer().GetPositions())
    result = Molecule.from_rdmol(mol)
    assert result.rdmol is mol
    assert result.smiles == Chem.MolToSmiles(mol)
    np.testing.assert_array_equal(result.conformer, expected)
    assert result.conformer.shape == (mol.GetNumAtoms(), 3)

@pytest.mark.parametrize('smiles', ['CCO', 'c1ccccc1'])
def test_missing_coordinates_can_be_generated_lazily(smiles):
    result = Molecule.from_rdmol(Chem.MolFromSmiles(smiles))
    assert result.conformer is None
    result._add_conformer(mode='2D')
    assert result.conformer.shape == (result.rdmol.GetNumAtoms(), 3)
    assert np.isfinite(result.conformer).all()
    np.testing.assert_array_equal(result.conformer[:, 2], 0)
    assert result.smiles == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
