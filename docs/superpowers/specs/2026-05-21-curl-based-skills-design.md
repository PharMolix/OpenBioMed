# Design: Curl-Based Skills for OpenBioMed API

## Summary

Convert the `iupac-name-identification-biot5` skill from Python code generation to curl POST requests against the OpenBioMed server API. This establishes a reusable pattern that other skills can follow.

## Motivation

Current skills instruct the agent to generate Python code using `open_biomed` imports and `TOOLS` registry. The project provides `run_pipeline` and `web_search` API endpoints that accomplish the same tasks via HTTP. Curl-based skills are simpler, don't require Python environment setup, and are more portable.

## Approach

Self-contained curl commands in each SKILL.md with a configurable `BASE_URL` variable. No shared files or helper scripts.

## Design

### SKILL.md Structure (for iupac-name-identification-biot5)

**Frontmatter**: Unchanged.

**Prerequisites section** (new):
- State that the OpenBioMed server must be running
- Define `BASE_URL` default as `http://localhost:8082` (user should adjust to their deployment)
- Server startup command: `sh ./scripts/run_server.sh` or individual uvicorn commands

**Workflow section** (replaced):
- Step 1 (optional): If user provides a molecule name (not SMILES), call `/web_search/` to get the SMILES string
- Step 2: Call `/run_pipeline/` with `molecule_question_answering` task using the SMILES and IUPAC question text

**Expected Outputs section** (updated):
- Document JSON response format for each API call with field descriptions

**Error Handling section** (updated):
- Server unreachable (connection refused)
- Empty/null response fields
- RDKit fallback as alternative approach (not error recovery)

**Example Usage section** (updated):
- Complete end-to-end example with actual curl commands and expected responses

### API Mapping

| Current Python Pattern | Curl Equivalent |
|---|---|
| `TOOLS["molecule_name_request"].run(accession="aspirin")` | `curl -X POST ${BASE_URL}/web_search/ -d '{"task": "molecule_name_request", "query": "aspirin"}'` |
| `TOOLS["molecule_question_answering"].run(molecule=mol, text=question)` | `curl -X POST ${BASE_URL}/run_pipeline/ -d '{"task": "molecule_question_answering", "model": "biot5", "molecule": "<SMILES>", "text": "What is the IUPAC name?"}'` |

### Response Formats

**`/web_search/` — molecule_name_request**:
```json
{
  "task": "molecule_name_request",
  "molecule": "<PubChem data>",
  "molecule_preview": "<SMILES string>"
}
```
Extract `molecule_preview` as the SMILES for the next step.

**`/run_pipeline/` — molecule_question_answering**:
```json
{
  "task": "molecule_question_answering",
  "model": "biot5",
  "text": "<IUPAC name answer>"
}
```
Extract `text` as the IUPAC name.

### Reusable Pattern for Other Skills

Any skill converting to curl follows this template:
1. Add Prerequisites section with `BASE_URL`
2. Replace each `TOOLS["..."]` call with the corresponding curl POST to `/run_pipeline/` or `/web_search/`
3. Document the JSON response format and which field to extract
4. Update error handling for HTTP-level failures
5. Provide a concrete example with curl commands and expected responses

### Files to Modify

- `skills/iupac-name-identification-biot5/SKILL.md` — full rewrite of Workflow, Expected Outputs, Error Handling, and Example Usage sections