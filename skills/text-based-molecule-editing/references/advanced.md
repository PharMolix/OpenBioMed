# Advanced Usage and Prompt Engineering

Tips for getting better results from text-based molecule editing.

## Prompt Engineering

### Effective Prompt Patterns

The model was trained on molecule-text pairs, so prompt style affects results.

#### Pattern 1: Property-Based Description

```
"This molecule should be more soluble in water"
"This molecule should have higher binding affinity to the target"
"This molecule should have better oral bioavailability"
```

#### Pattern 2: Functional Description

```
"This molecule can bind with recombinant human 15-LOX-1"
"This molecule is a selective inhibitor of COX-2"
"This molecule is a substrate for P-glycoprotein"
```

#### Pattern 3: Structural Hints

```
"Replace the phenyl ring with a pyridine"
"Add a hydroxyl group to improve solubility"
"Remove the methyl group to reduce lipophilicity"
```

### Prompt Quality Guidelines

| Good Prompt | Poor Prompt | Why |
|-------------|-------------|-----|
| "This molecule should be more soluble in water" | "make soluble" | Complete sentence matches training data |
| "This molecule should have improved drug-likeness" | "better drug" | Clear property specification |
| "This molecule can bind to the serotonin receptor" | "binds stuff" | Specific biological target |

### Combining Multiple Properties

The model can handle multi-objective prompts:

```python
# Combined property prompt
text = "This molecule should be more soluble and have better drug-likeness"

# Prioritized properties
text = "This molecule should have higher binding affinity while maintaining good solubility"
```

Note: Results may trade off between properties. Run multiple times to explore the Pareto front.

## Advanced Workflows

### Iterative Optimization

Run multiple editing cycles to reach target properties:

```python
from open_biomed.data import Molecule, Text
from open_biomed.core.pipeline import InferencePipeline

def iterative_optimization(molecule, prompts, max_iterations=5, target_logp=1.0):
    """
    Iteratively edit molecule until target property is reached.

    Args:
        molecule: Starting molecule
        prompts: List of edit prompts to cycle through
        max_iterations: Maximum number of editing cycles
        target_logp: Target LogP value

    Returns:
        Optimized molecule
    """
    pipeline = InferencePipeline(
        task="text_based_molecule_editing",
        model="molt5",
        model_ckpt="./checkpoints/server/text_based_molecule_editing_biot5.ckpt",
        device="cuda:0"
    )

    logp_tool = TOOLS["molecule_logp"]
    current_mol = molecule

    for i in range(max_iterations):
        prompt = prompts[i % len(prompts)]
        outputs = pipeline.run(
            molecule=current_mol,
            text=Text.from_str(prompt),
        )
        new_mol = outputs[0][0]

        if new_mol is None:
            continue

        logp, _ = logp_tool.run(molecule=new_mol)

        if logp[0] <= target_logp:
            print(f"Target reached after {i+1} iterations")
            return new_mol

        current_mol = new_mol

    return current_mol
```

### Batch Processing

Process multiple molecules efficiently:

```python
def batch_edit(molecules, prompt, device="cuda:0"):
    """
    Edit multiple molecules with the same prompt.

    Args:
        molecules: List of Molecule objects
        prompt: Edit prompt string
        device: Device for inference

    Returns:
        List of edited molecules
    """
    pipeline = InferencePipeline(
        task="text_based_molecule_editing",
        model="molt5",
        model_ckpt="./checkpoints/server/text_based_molecule_editing_biot5.ckpt",
        device=device
    )

    text = Text.from_str(prompt)
    edited_mols = []

    for mol in molecules:
        outputs = pipeline.run(molecule=mol, text=text)
        edited_mols.append(outputs[0][0] if outputs[0] else None)

    return edited_mols
```

### Diversity-Driven Generation

Generate diverse candidates by varying prompts:

```python
def generate_diverse_edits(molecule, base_property="soluble"):
    """
    Generate diverse molecule edits for a property.

    Args:
        molecule: Input molecule
        base_property: Target property

    Returns:
        List of diverse edited molecules
    """
    prompts = [
        f"This molecule should be more {base_property}",
        f"This molecule should have improved {base_property}",
        f"This molecule is highly {base_property}",
        f"Increase the {base_property} of this molecule",
    ]

    pipeline = InferencePipeline(
        task="text_based_molecule_editing",
        model="molt5",
        model_ckpt="./checkpoints/server/text_based_molecule_editing_biot5.ckpt",
        device="cuda:0"
    )

    results = []
    for prompt in prompts:
        outputs = pipeline.run(
            molecule=molecule,
            text=Text.from_str(prompt),
        )
        if outputs[0][0] is not None:
            results.append(outputs[0][0])

    # Deduplicate by SMILES
    unique_smiles = set()
    unique_mols = []
    for mol in results:
        if mol.smiles not in unique_smiles:
            unique_smiles.add(mol.smiles)
            unique_mols.append(mol)

    return unique_mols
```

## Integration with Other Tools

### Combine with ADMET Prediction

```python
from open_biomed.tools.tool_registry import TOOLS

def edit_with_admet_check(molecule, edit_prompt, target_profile):
    """
    Edit molecule and verify ADMET properties.

    Args:
        molecule: Input molecule
        edit_prompt: Editing instruction
        target_profile: Dict of target property ranges

    Returns:
        Tuple of (edited_molecule, passes_filters)
    """
    # Step 1: Edit molecule
    pipeline = InferencePipeline(
        task="text_based_molecule_editing",
        model="molt5",
        model_ckpt="./checkpoints/server/text_based_molecule_editing_biot5.ckpt",
        device="cuda:0"
    )

    outputs = pipeline.run(
        molecule=molecule,
        text=Text.from_str(edit_prompt),
    )
    edited = outputs[0][0]

    if edited is None:
        return None, False

    # Step 2: Check ADMET properties
    qed_tool = TOOLS["molecule_qed"]
    logp_tool = TOOLS["molecule_logp"]

    qed, _ = qed_tool.run(molecule=edited)
    logp, _ = logp_tool.run(molecule=edited)

    passes = True
    if "qed_min" in target_profile and qed[0] < target_profile["qed_min"]:
        passes = False
    if "logp_max" in target_profile and logp[0] > target_profile["logp_max"]:
        passes = False

    return edited, passes
```

### Combine with Docking

```python
def edit_for_binding(molecule, protein, binding_prompt):
    """
    Edit molecule for improved binding, validated by docking.

    Args:
        molecule: Input molecule
        protein: Target protein
        binding_prompt: Binding-related edit prompt

    Returns:
        Edited molecule if binding improved, else original
    """
    from open_biomed.tasks.aidd_tasks.protein_molecule_docking import VinaDockTask

    # Get baseline docking score
    vina = VinaDockTask()
    baseline_score = vina.run(molecule=molecule, protein=protein)[0][0]

    # Edit molecule
    pipeline = InferencePipeline(
        task="text_based_molecule_editing",
        model="molt5",
        model_ckpt="./checkpoints/server/text_based_molecule_editing_biot5.ckpt",
        device="cuda:0"
    )

    outputs = pipeline.run(
        molecule=molecule,
        text=Text.from_str(binding_prompt),
    )
    edited = outputs[0][0]

    if edited is None:
        return molecule

    # Dock edited molecule
    new_score = vina.run(molecule=edited, protein=protein)[0][0]

    # Return better molecule (more negative = better binding)
    if new_score < baseline_score:
        print(f"Binding improved: {baseline_score:.2f} → {new_score:.2f}")
        return edited
    else:
        print(f"No improvement: {baseline_score:.2f} vs {new_score:.2f}")
        return molecule
```

## Model Selection

### MolT5 vs BioT5

| Model | Strengths | Best For |
|-------|-----------|----------|
| MolT5 | SMILES generation quality | Structure-focused edits |
| BioT5 | Biology knowledge integration | Target-focused edits |
| BioT5+ | Enhanced multi-modal | Complex property edits |

The checkpoint `text_based_molecule_editing_biot5.ckpt` was trained on BioT5 architecture but can be loaded with MolT5 base if BioT5 base is unavailable.

### Performance Comparison

| Model | Valid SMILES Rate | Property Improvement |
|-------|-------------------|---------------------|
| MolT5 | ~95% | Good for simple properties |
| BioT5 | ~93% | Better for bioactivity |
| BioT5+ | ~91% | Best for complex prompts |
