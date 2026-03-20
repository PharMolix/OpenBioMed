# Custom Question Templates

The molecule_question_answering tool can answer various types of questions about molecules. Here are some useful question templates:

## Biochemical and Functional Questions

| Question Template | Purpose |
|-------------------|---------|
| "I am interested in understanding the molecule biochemical significance; can you describe its roles in biology and chemistry?" | General biochemical significance (default) |
| "What is the primary biological function of this molecule?" | Primary function |
| "What are the therapeutic applications of this molecule?" | Medical/therapeutic uses |
| "Is this molecule a natural product or synthetic?" | Origin classification |
| "What metabolic pathways involve this molecule?" | Metabolic context |

## Chemical Property Questions

| Question Template | Purpose |
|-------------------|---------|
| "Could you provide the systematic name of this compound according to IUPAC nomenclature?" | IUPAC name |
| "What are the key functional groups in this molecule?" | Structural features |
| "Describe the solubility profile of this molecule." | Solubility properties |
| "What is the molecular weight of this compound?" | Molecular weight |

## Drug Discovery Questions

| Question Template | Purpose |
|-------------------|---------|
| "What are the known drug targets of this molecule?" | Target information |
| "What are the potential side effects of this molecule?" | Safety profile |
| "Is this molecule suitable for oral administration?" | Drug-likeness |
| "What diseases or conditions is this molecule used to treat?" | Therapeutic indications |

## Example Usage

```python
from open_biomed.data import Molecule, Text
from open_biomed.tools.tool_registry import TOOLS

molecule = Molecule.from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin
qa_tool = TOOLS["molecule_question_answering"]

# Ask about IUPAC name
question = Text.from_str("Could you provide the systematic name of this compound according to IUPAC nomenclature?")
outputs, _ = qa_tool.run(molecule=molecule, text=question)
print(outputs[0])

# Ask about therapeutic applications
question = Text.from_str("What are the therapeutic applications of this molecule?")
outputs, _ = qa_tool.run(molecule=molecule, text=question)
print(outputs[0])
```

## Notes

- Questions should be clear and specific
- The model performs best on questions about common molecules
- Complex questions may yield more detailed but less precise answers
- The model may not have knowledge of very rare or recently discovered molecules
