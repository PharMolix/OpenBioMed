---
name: molecule-biochemical-significance-query-biot5
description: >
  Query a molecule's biochemical significance and roles in biology and chemistry
  using BioT5 multi-modal model.
  Use this skill when:
  (1) Understanding a molecule's biological roles and functions,
  (2) Describing a molecule's chemical significance and applications,
  (3) Getting natural language explanations of molecular properties,
  (4) Summarizing what a molecule is used for or its metabolic relevance.
license: MIT
category: multi-modal-reasoning
tags: [molecule, question-answering, biochemistry, biot5, multi-modal]
---

# Molecule Biochemical Significance Query

Query a molecule's biochemical significance using BioT5 multi-modal model.

## When to Use

- User asks about a molecule's biological roles or functions
- User wants to understand what a molecule is used for
- User requests natural language description of molecular properties
- User asks about a molecule's metabolic or chemical significance

## Workflow

### Step 1: Create Molecule

Create molecule from SMILES string.

```python
from open_biomed.data import Molecule
molecule = Molecule.from_smiles("CCCCCCCc1ccco1")  # Heptylfuran
```

### Step 2: Ask About Biochemical Significance

Use the molecule_question_answering tool with the default question.

```python
from open_biomed.data import Text
from open_biomed.tools.tool_registry import TOOLS

qa_tool = TOOLS["molecule_question_answering"]
question = Text.from_str(
    "I am interested in understanding the molecule biochemical significance; "
    "can you describe its roles in biology and chemistry?"
)
outputs, _ = qa_tool.run(molecule=molecule, text=question)
print(outputs[0])  # Natural language answer
```

### Alternative: Get Molecule by Name

If you have a molecule name instead of SMILES:

```python
from open_biomed.tools.tool_registry import TOOLS

# Get molecule from name
name_tool = TOOLS["molecule_name_request"]
molecules, _ = name_tool.run("aspirin")
molecule = molecules[0]

# Then proceed with QA
qa_tool = TOOLS["molecule_question_answering"]
question = Text.from_str(
    "I am interested in understanding the molecule biochemical significance; "
    "can you describe its roles in biology and chemistry?"
)
outputs, _ = qa_tool.run(molecule=molecule, text=question)
```

## Expected Outputs

| Output | Description |
|--------|-------------|
| Natural language text | Description of biochemical roles, applications, and significance |

### Example Outputs

| Molecule | Output |
|----------|--------|
| `CCCCCCCc1ccco1` (heptylfuran) | "flavouring agent; fragrance; metabolite" |
| `CC(=O)OC1=CC=CC=C1C(=O)O` (aspirin) | "analgesic; anti-inflammatory; antipyretic" |

## Model Details

- **Model**: BioT5
- **Checkpoint**: `./checkpoints/server/molecule_question_answering_biot5.ckpt`
- **Device**: CUDA (default: cuda:0)

## Error Handling

| Error | Solution |
|-------|----------|
| Checkpoint not found | Ensure `./checkpoints/server/molecule_question_answering_biot5.ckpt` exists |
| Invalid SMILES | Validate SMILES format or use molecule name lookup |
| CUDA out of memory | Set device to "cpu" in the tool configuration |

## See Also

- `examples/basic_example.py` - Complete runnable example
- `references/custom_questions.md` - Other question templates
