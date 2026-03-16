# Pocket-Based Drug Design Skill

This skill implements structure-based drug design using MolCraft to generate novel molecules that fit protein binding pockets, based on the successful execution from the OpenBioMed logs (4xli kinase inhibitor design).

## What We've Accomplished

### 1. Core Implementation
- ✅ **Protein Retrieval**: Fetch structures from PDB database
- ✅ **Molecule Extraction**: Extract ligands to define binding pockets
- ✅ **Property Calculation**: QED, SA, LogP, Lipinski rules
- ✅ **Similarity Analysis**: Tanimoto similarity for diversity control
- ✅ **Tool Integration**: All core OpenBioMed tools functional

### 2. Supported Capabilities
The skill supports molecule generation for proteins with defined binding pockets:
- **Basic Usage**: PDB ID, number of candidates, similarity threshold
- **Advanced Options**: Pocket residues, property constraints, visualization

### 3. Test Results
All tests pass successfully:
- ✅ Basic Workflow Test
- ✅ Target Adaptation Test
- ✅ Skill Interface Test

## Files Created

1. **SKILL.md** - Comprehensive documentation with:
   - Core implementation steps and code
   - Target adaptation guidelines
   - Usage examples
   - Performance metrics

2. **test_skill.py** - Complete test suite that:
   - Validates tool functionality
   - Tests configuration interfaces
   - Demonstrates target adaptation

3. **README.md** - This summary file

## Usage

### Basic Usage
```python
# Generate 10 diverse inhibitors for 4xli
skill: pocket_based_drug_design
args:
  protein_id: "4xli"
  num_candidates: 10
  similarity_threshold: 0.7
```

### For New Targets
```python
# Configure for protease target
target_config = {
    "protein_id": "1hpv",
    "pocket_source": "residues",
    "binding_site_residues": [25, 26, 27, 28, 29],
    "property_focus": "bioavailability"
}
```

## Key Features

1. **Modular Design** - Each step can be customized or replaced
2. **Extensible** - Easy to add new evaluation criteria
3. **Robust** - Handles various protein structures and ligands
4. **Configurable** - Flexible parameters for different use cases

## Next Steps

The skill is ready for production use. Future enhancements could include:
- Integration with more advanced scoring functions
- Batch processing for multiple targets
- Property constraint optimization algorithms
- Visualization customization options