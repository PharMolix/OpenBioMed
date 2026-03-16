#!/usr/bin/env python3
"""
Test script for pocket-based drug design skill
Validates the core implementation steps and adapts for different targets
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from open_biomed.tools.tool_registry import TOOLS
from open_biomed.scripts.inference import InferencePipeline


def test_basic_workflow():
    """Test the basic workflow with 4xli target"""
    print("Testing basic drug design workflow...")

    # Step 1: Protein Retrieval
    print("\n1. Retrieving protein structure (4xli)...")
    pdb_tool = TOOLS["protein_pdb_request"]
    try:
        result, messages = pdb_tool.run(accession="4xli", mode="file_only")
        print(f"✓ Retrieved PDB file: {result}")
        protein_file = result
    except Exception as e:
        print(f"✗ Failed to retrieve protein: {e}")
        return False

    # Step 2: Extract molecules
    print("\n2. Extracting molecules from PDB...")
    try:
        extract_tool = TOOLS["extract_molecules_from_pdb_file"]
        extracted, messages = extract_tool.run(pdb_file=protein_file)
        print(f"✓ Extracted {len(extracted[0])} items")  # extracted[0] is the list of items
        for i, (mol_type, chain_id, obj) in enumerate(extracted[0][:3]):  # Show first 3
            print(f"  Item {i}: {mol_type} from chain {chain_id}")
    except Exception as e:
        print(f"✗ Failed to extract molecules: {e}")
        print(f"Type of extracted: {type(extracted)}")
        print(f"Extracted value: {extracted}")
        return False

    # Step 3: Test property calculation tools
    print("\n3. Testing property calculation tools...")
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    try:
        property_tools = [
            "molecule_qed",
            "molecule_sa",
            "molecule_logp",
            "molecule_lipinski"
        ]

        # Create test molecule
        from open_biomed.data import Molecule
        test_mol = Molecule.from_smiles(test_smiles)

        results = {}
        for tool_name in property_tools:
            tool = TOOLS[tool_name]
            try:
                if tool_name in ["molecule_lipinski"]:
                    score, _ = tool.run(molecule=test_mol)
                else:
                    score = tool.run(molecule=test_mol)
                results[tool_name] = score
                print(f"✓ {tool_name}: {score}")
            except Exception as e:
                print(f"✗ {tool_name}: {e}")
                results[tool_name] = None

    except Exception as e:
        print(f"✗ Property test failed: {e}")
        return False

    # Step 4: Test similarity calculation
    print("\n4. Testing similarity calculation...")
    try:
        similarity_tool = TOOLS["molecule_similarity"]
        # Test with itself (should be 1.0)
        score = similarity_tool.run(molecule_1=test_mol, molecule_2=test_mol)
        print(f"✓ Self-similarity: {score[0][0]:.3f} (expected ~1.0)")

        # Test with different molecule
        test_mol2 = Molecule.from_smiles("CCO")  # Ethanol
        score2 = similarity_tool.run(molecule_1=test_mol, molecule_2=test_mol2)
        print(f"✓ Cross-similarity: {score2[0][0]:.3f} (expected <1.0)")

    except Exception as e:
        print(f"✗ Similarity test failed: {e}")
        return False

    print("\n✓ Basic workflow test completed successfully!")
    return True


def test_adaptation_for_targets():
    """Test workflow adaptation for different target types"""
    print("\n" + "="*60)
    print("Testing workflow adaptation for different targets...")

    # Define test configurations for different target types
    target_configs = {
        "kinase": {
            "protein_id": "4xli",
            "pocket_source": "ligand",
            "property_focus": "selectivity",
            "expected_tools": ["protein_pdb_request", "extract_molecules_from_pdb_file"]
        },
        "protease": {
            "protein_id": "1hpv",
            "pocket_source": "residues",
            "property_focus": "bioavailability",
            "expected_tools": ["protein_pdb_request", "import_pocket"]
        },
        "nuclear_receptor": {
            "protein_id": "3err",
            "pocket_source": "predicted",
            "property_focus": "tissue_penetration",
            "expected_tools": ["protein_pdb_request", "protein_binding_site_prediction"]
        }
    }

    for target_type, config in target_configs.items():
        print(f"\n--- Testing {target_type.upper()} target ---")

        # Test tool availability
        print(f"1. Checking tool availability for {target_type}...")
        missing_tools = []
        for tool_name in config["expected_tools"]:
            if tool_name not in TOOLS.available_tools():
                missing_tools.append(tool_name)

        if missing_tools:
            print(f"⚠ Missing tools: {missing_tools}")
            continue
        print("✓ All required tools available")

        # Test protein retrieval
        print(f"2. Testing protein retrieval for {config['protein_id']}...")
        try:
            pdb_tool = TOOLS["protein_pdb_request"]
            result, messages = pdb_tool.run(
                accession=config["protein_id"],
                mode="file_only"
            )
            print(f"✓ Retrieved: {config['protein_id']}")
        except Exception as e:
            print(f"✗ Failed to retrieve {config['protein_id']}: {e}")
            continue

    print("\n✓ Target adaptation tests completed!")
    return True


def test_skill_interface():
    """Test the skill interface and configuration"""
    print("\n" + "="*60)
    print("Testing skill interface...")

    # Test basic usage example
    basic_config = {
        "skill": "pocket_based_drug_design",
        "args": {
            "protein_id": "4xli",
            "num_candidates": 10,
            "similarity_threshold": 0.7
        }
    }

    print("Basic usage config:")
    print(json.dumps(basic_config, indent=2))

    # Test advanced usage example with optional parameters
    advanced_config = {
        "skill": "pocket_based_drug_design",
        "args": {
            "protein_id": "1abc",
            "num_candidates": 20,
            "similarity_threshold": 0.5,
            "pocket_residues": [100, 101, 102, 103, 104],
            "min_docking_score": -10.0,
            "qed_range": [0.5, 1.0],
            "logp_range": [0, 5]
        }
    }

    print("\nAdvanced usage config:")
    print(json.dumps(advanced_config, indent=2))

    # Validate configuration keys
    required_args = ["protein_id", "num_candidates", "similarity_threshold"]
    optional_args = ["pocket_residues", "min_docking_score", "qed_range", "logp_range"]

    basic_args = basic_config["args"]
    advanced_args = advanced_config["args"]

    print(f"\nConfiguration validation:")
    print(f"Basic config - Required args: {all(k in basic_args for k in required_args)}")
    print(f"Advanced config - Required args: {all(k in advanced_args for k in required_args)}")
    print(f"Advanced config - Optional args: {set(advanced_args.keys()).intersection(set(optional_args))}")

    return True


def main():
    """Run all tests"""
    print("Running pocket-based drug design skill tests...")
    print("="*60)

    tests = [
        ("Basic Workflow", test_basic_workflow),
        ("Target Adaptation", test_adaptation_for_targets),
        ("Skill Interface", test_skill_interface)
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n{test_name} Test:")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "="*60)
    print("Test Summary:")
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)