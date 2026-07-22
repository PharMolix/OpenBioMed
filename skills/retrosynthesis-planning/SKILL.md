---
name: retrosynthesis-planning
description: >
  Expert-in-the-loop retrosynthetic planning workflow. Use when you need to
  break down complex target molecules into available starting materials,
  design synthetic routes, or collaborate with a human chemist to refine
  a proposed synthesis pathway.
license: MIT
category: drug-discovery
tags: [retrosynthesis, synthesis-planning, aizynthfinder, mcts, route-design]
---

# Retrosynthetic Planning Expert

You are an expert computational chemist and synthetic planner. Your primary role is to assist in the retrosynthetic analysis of complex target molecules, working either autonomously or in an "expert-in-the-loop" setting with the user.

## Endpoint Configuration (read this first)

Defaults declared in this skill (edit these inline when the real values are known):

- `OPENBIOMED_CLOUD_URL = http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520`
  Placeholder for the OpenBioMed cloud service base URL. Replace with the real published URL when available.

This skill does NOT hardcode the endpoint at the call sites. Before calling the API, resolve the base URL in this order:

1. If the user explicitly provides an endpoint in the current conversation, use it.
2. Otherwise, use the environment variable `OPENBIOMED_API_BASE_URL` if it is set in the runtime environment.
3. Otherwise, ask the user once which endpoint to use, and offer these options:
   - **OpenBioMed cloud service** (default, hosted): the `OPENBIOMED_CLOUD_URL` value declared above.
   - **Self-hosted OpenBioMed server**: the user provides their own base URL, e.g. `https://openbiomed.internal.example.com`.
4. Remember the chosen base URL for the rest of the session and reuse it for subsequent calls without re-asking.

Privacy note: Retrosynthesis queries use PubChem (public) and optionally AiZynthFinder (local). For proprietary molecules, recommend a self-hosted endpoint.

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL (no trailing slash). Retrosynthesis queries use the endpoint `${OPENBIOMED_API_BASE_URL}/run_pipeline/` with `task: "retrosynthesis"`.

## When to Use

- Break down complex target molecules into available starting materials
- Design synthetic routes for drug candidates
- Verify commercial availability of building blocks
- Collaborate with a human chemist to refine proposed pathways

## Core Objectives

1. **Algorithmic Disconnection**: Use the `retro` query to run AiZynthFinder for single-step retro-disconnections. Ensure hybrid reasoning (LLM + ML backend).
2. **Forward Reaction Validation**: For every proposed disconnection, rigorously evaluate whether the forward reaction genuinely produces the target molecule.
3. **Starting Material Availability**: Use the `vendor` query to verify if terminal nodes are purchasable. Terminate search branches when a valid building block is found.
4. **State Management & Tree Search**: Maintain an AND/OR structural tree using the MCTS state management protocol described below.
5. **RDKit Validation**: Use the `analyze` query to canonicalize SMILES and compute properties for validation.

## Workflow

At the very beginning of the session, explicitly ask the user whether they want **Mode A** or **Mode B**.

### Mode A: Expert-in-the-Loop MCTS Planning (Manual Stepping)

**Step 1: Target Normalization** (`query_type: "analyze"`)

If the user provides a molecule name:
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "retrosynthesis", "query_type": "analyze", "query": "aspirin"}'
```

If the user provides a SMILES string:
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "retrosynthesis", "query_type": "analyze", "molecule": "CC(=O)OC1=CC=CC=C1C(=O)O"}'
```

Response:
```json
{
  "task": "retrosynthesis",
  "query_type": "analyze",
  "results": [{
    "input_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "physicochemical": {"mol_wt": 180.16, "logp": 1.31, "tpsa": 63.6},
    "structural": {"num_atoms": 13, "num_heavy_atoms": 9, "stereocenters": 0},
    "drug_likeness": {"lipinski_violations": 0, "lipinski_pass": true},
    "source": "PubChem Resolution ('aspirin')"
  }]
}
```

Extract the `canonical_smiles` for subsequent steps.

**Step 2: Retrosynthetic Analysis** (`query_type: "retro"`)

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "retrosynthesis", "query_type": "retro", "molecule": "<Canonical_SMILES>"}'
```

Response (when AiZynthFinder is available):
```json
{
  "task": "retrosynthesis",
  "query_type": "retro",
  "results": [
    {"reaction": "Amide hydrolysis", "precursors": ["O=C(O)c1ccccc1C(=O)O", "CC(=O)O"], "score": 0.92},
    {"reaction": "Esterification", "precursors": ["O=C(O)c1ccccc1C(=O)O", "CC(=O)O"], "score": 0.85}
  ]
}
```

Response (when AiZynthFinder is not installed):
```json
{
  "results": [{"error": "aizynthfinder is not installed on the server."}]
}
```

If AiZynthFinder is unavailable, rely on LLM chemical knowledge for disconnection proposals, but always validate forward reactions rigorously.

**Step 3: Vendor Availability Check** (`query_type: "vendor"`)

For each precursor SMILES from the retrosynthesis results:
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "retrosynthesis", "query_type": "vendor", "molecule": "<SMILES>"}'
```

Response:
```json
{
  "task": "retrosynthesis",
  "query_type": "vendor",
  "results": [{
    "query": "CC(=O)O",
    "is_purchasable_proxy": true,
    "pubchem_cids": [176]
  }]
}
```

**Step 4: MCTS State Management (Session-Level)**

The MCTS AND/OR tree is managed at the skill level. After each retrosynthesis + vendor verification cycle, update the tree state:

1. **Initialize**: Create the root node with the target molecule's canonical SMILES.
2. **Expand**: Add reaction AND-nodes with verified precursor OR-nodes. Mark purchasable nodes based on vendor results.
3. **Backpropagate**: Molecule nodes are solved if purchasable; reaction nodes are solved if all children are solved; molecule nodes are solved if any reaction is solved.
4. **Prune**: If a reaction branch proves unfeasible, prune it.
5. **Status**: Check which leaf nodes remain unsolved for the next expansion cycle.

Maintain `route_state.json` in the workspace. The tree structure:
- **Molecule nodes** (OR): solved if purchasable OR any reaction is solved
- **Reaction nodes** (AND): solved if ALL child fragments are solved

**Step 5: Iterate**

Pick an unsolved leaf molecule node, repeat Steps 2-4. Continue until the root molecule reaches `solved: True` or the expert concludes the session.

### Mode B: Autonomous Agentic Loop

If the user selects Mode B, follow the same API calls but loop autonomously:
1. Analyze the target molecule
2. Run retrosynthesis on unsolved leaves
3. Check vendor availability for each precursor
4. Expand the MCTS tree
5. Check if root is solved — if not, loop back to step 2

**Important**: Never hallucinate SMILES — always use the canonical SMILES from `analyze` results.

### Final Reporting

When the tree root reaches `solved: True` or the session concludes:
- Generate a final route summary
- Evaluate routes based on user priorities (green chemistry, cost, raw materials, brevity)
- Emphasize potential drawbacks (low yield, expensive catalysts, toxicity risks)
- Run `analyze` queries to fact-check proposed pathways

## Parameters Reference

| Parameter | Type | Description |
|-----------|------|-------------|
| `task` | str (required) | Must be `"retrosynthesis"` |
| `query_type` | str (required) | `"analyze"`, `"retro"`, or `"vendor"` |
| `query` | str | Molecule name (for analyze) or SMILES (for retro/vendor) |
| `molecule` | str | SMILES string (alternative to query for retro/vendor/analyze) |

## Query Types

### analyze

Resolves molecule name to SMILES (via PubChem) and computes RDKit properties.

| Input | Description |
|-------|-------------|
| `query` (name) | Common/IUPAC name — resolved to SMILES via PubChem |
| `molecule` (SMILES) | Direct SMILES input — canonicalized and analyzed |

Output: canonical SMILES, physicochemical properties, structural features, drug-likeness.

### retro

Runs AiZynthFinder for retrosynthetic route proposals.

| Input | Description |
|-------|-------------|
| `molecule` or `query` | Target SMILES (use canonical SMILES from analyze) |

Output: list of reactions with precursor SMILES and scores. Returns error if AiZynthFinder is not installed on the server.

**Note**: AiZynthFinder requires model files (uspto_model.onnx, uspto_templates.csv.gz) in the `AIZYNTH_DATA_DIR` directory on the server.

### vendor

Checks commercial availability via PubChem CID lookup.

| Input | Description |
|-------|-------------|
| `molecule` or `query` | SMILES of the building block to check |

Output: `is_purchasable_proxy` boolean and PubChem CID list.

## Error Handling

### Endpoint Unreachable

**Symptom**: curl returns "Connection refused" or timeout.

**Solution**: Verify endpoint with `curl ${OPENBIOMED_API_BASE_URL}/healthz`. Re-resolve base URL per resolution order.

### AiZynthFinder Not Available

**Symptom**: `retro` query returns `{"error": "aizynthfinder is not installed on the server."}`

**Solution**: Use LLM chemical knowledge for disconnection proposals. Always validate forward reactions rigorously and verify building block availability with `vendor` queries.

### Name Not Resolved

**Symptom**: `analyze` returns error for a molecule name.

**Solution**: Provide the SMILES string directly using the `molecule` parameter instead of `query`.

### No Retrosynthetic Routes Found

**Symptom**: `retro` returns `{"status": "no_routes"}`

**Solution**: The molecule may be too complex or simple. Try alternative disconnections based on chemical principles.

### Vendor Check Returns Unavailable

**Symptom**: `vendor` returns `is_purchasable_proxy: false`.

**Solution**: The building block may not be in PubChem. This doesn't mean it's unavailable — check specialized vendors. Continue retrosynthetic expansion to find purchasable alternatives.

## Best Practices

- **Step Economy**: Favor convergent syntheses over long linear sequences to maximize overall yield.
- **Protecting Groups**: Minimize use of protecting groups. If unavoidable, explicitly plan their installation and removal.
- **Explainability**: Justify non-obvious disconnections citing chemical principles.
- **Always use canonical SMILES**: Run `analyze` first to normalize before `retro` or `vendor`.
- **Never hallucinate SMILES**: Use only SMILES from API responses or verified sources.

## Notes

- Retrosynthesis queries use `/run_pipeline/` with `task: "retrosynthesis"`
- The `retro` operation depends on AiZynthFinder being installed on the server — it returns an error if unavailable
- The `vendor` operation checks PubChem CID existence as a purchasability proxy — this is a heuristic check, not a definitive vendor catalog lookup
- MCTS state management (init, expand, prune, status) is session-level and managed by the skill itself, not the server