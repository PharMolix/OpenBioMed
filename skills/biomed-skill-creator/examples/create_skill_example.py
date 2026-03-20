"""
Example: Creating a New BioMed Skill

This demonstrates the interactive workflow of the biomed-skill-creator meta-skill.
"""

# ============================================================
# SCENARIO: User wants to create a skill for analyzing protein mutations
# ============================================================

# Step 1: CAPTURE INTENT
# ============================================================
# User says: "I want a skill that helps me understand how mutations affect proteins"
#
# Questions to ask:
# - What inputs? (protein ID, mutation notation like "A123V")
# - What outputs? (explanation of effects, recommendations)
# - What tools? (mutation_explanation, mutation_engineering)

# Step 2: DESIGN WORKFLOW
# ============================================================
# Proposed workflow:
#
# 1. Get protein (from UniProt ID or FASTA)
# 2. Apply mutation notation
# 3. Explain mutation effects using MutaPLM
# 4. Summarize findings

PROPOSED_WORKFLOW = """
from open_biomed.tools import TOOLS
from open_biomed.data import Protein

# Step 1: Get protein
tool = TOOLS["protein_uniprot_request"]
result, msg = tool.run(accession="P00533")  # EGFR
protein = result["protein"]

# Step 2: Explain mutation
from open_biomed.core.pipeline import InferencePipeline
pipeline = InferencePipeline(
    task="mutation_explanation",
    model="mutaplm",
    device="cuda:0"
)
explanation = pipeline.run(protein=protein, mutation="L858R")

# Step 3: Summarize
print(f"Mutation L858R in EGFR:")
print(explanation)
"""

# Step 3: INTERACTIVE VALIDATION
# ============================================================
# Ask user: "Please provide a protein ID and mutation to test this workflow"
#
# User provides: "P00533" (EGFR) and mutation "L858R"
#
# Execute and show results:

def demonstrate_validation():
    """
    Shows how validation would be displayed to user.
    """
    print("=" * 60)
    print("INTERACTIVE VALIDATION")
    print("=" * 60)

    print("\n=== Step 1: Retrieving Protein ===")
    print("Tool: protein_uniprot_request")
    print("Input: accession='P00533'")
    print("-" * 40)
    # Actual execution would happen here
    print("Result: Protein retrieved successfully")
    print("  - Name: Epidermal growth factor receptor (EGFR)")
    print("  - Length: 1210 amino acids")
    print("  - UniProt ID: P00533")

    print("\n=== Step 2: Explaining Mutation ===")
    print("Tool: mutation_explanation (MutaPLM)")
    print("Input: protein=EGFR, mutation='L858R'")
    print("-" * 40)
    print("Result: Mutation effect predicted")
    print("  - L858R is a common activating mutation in EGFR")
    print("  - Associated with increased kinase activity")
    print("  - Often found in non-small cell lung cancer")
    print("  - Sensitive to EGFR tyrosine kinase inhibitors")

    print("\n=== Step 3: Summary ===")
    print("The L858R mutation in EGFR is a well-characterized")
    print("activating mutation with clinical significance for")
    print("lung cancer treatment decisions.")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print("\nThe workflow executed successfully.")
    print("\nDo you want to:")
    print("1. Proceed with this workflow?")
    print("2. Modify a step?")
    print("3. Try a different input?")


# Step 4: FINALIZE SKILL
# ============================================================
# After user approves, create SKILL.md

SKILL_TEMPLATE = """
---
name: protein-mutation-analyzer
description: |
  Analyze the effects of protein mutations and predict their impact.
  Use this skill when the user asks about mutation effects, variant
  interpretation, or wants to understand how changes affect protein
  function. Triggers on "analyze mutation", "variant effect", "what
  does this mutation do", "interpret this variant".
---

# Protein Mutation Analyzer

## When to Use
- User provides a protein ID and mutation notation
- User asks about effects of specific mutations
- User wants variant interpretation for clinical or research purposes

## Workflow

### Step 1: Get the Protein

```python
from open_biomed.tools import TOOLS

tool = TOOLS["protein_uniprot_request"]
result, message = tool.run(accession="P00533")  # UniProt ID
protein = result["protein"]
print(f"Retrieved: {protein.name}")
```

### Step 2: Explain the Mutation

```python
from open_biomed.core.pipeline import InferencePipeline

pipeline = InferencePipeline(
    task="mutation_explanation",
    model="mutaplm",
    device="cuda:0"
)

# Mutation notation: OriginalAA_Position_NewAA
explanation = pipeline.run(protein=protein, mutation="L858R")
```

### Step 3: Summarize Findings

Present structured output:
- Mutation location and type
- Predicted effects
- Clinical relevance (if known)
- Recommendations

## Mutation Notation

Use standard notation: `{OriginalAA}{Position}{NewAA}`
- Example: `L858R` = Leucine at position 858 changed to Arginine
- Example: `T790M` = Threonine at position 790 changed to Methionine

## Error Handling

- If protein not found in UniProt, ask for FASTA sequence
- If mutation position out of range, notify user
- If model unavailable, provide general interpretation guidance
"""


if __name__ == "__main__":
    print("BioMed Skill Creator - Demonstration")
    print("=" * 60)
    print()
    print("This example shows the interactive validation process")
    print("for creating biomedical skills.")
    print()
    demonstrate_validation()
