---
name: disease-drug-intelligence
description: >
  Disease-to-innovative-drug comprehensive analysis skill for biomedical Q&A scenarios.
  Used to answer questions like "What innovative/frontier/in-development/new-mechanism drugs are available for a disease?",
  outputting integrated evidence reports on disease-target-drug-clinical progress-mechanism trends.
  Suitable for multi-database queries (ChEMBL, ClinicalTrials, Tavily Search), deduplication, innovation screening, and structured report generation.
license: MIT
---

# Disease-Drug Intelligence Integration

Converts natural language questions (e.g., "What noteworthy new drugs are available for Alzheimer's disease recently?") into executable multi-database query plans.
Generates decision-oriented comprehensive reports in Chinese, not just a list of drug names.

## Endpoint Configuration (read this first)

Defaults declared in this skill (edit these inline when the real values are known):

- `OPENBIOMED_CLOUD_URL = http://127.0.0.1:8092`
  Placeholder for the OpenBioMed cloud service base URL. Replace with the real published URL when available.

This skill does NOT hardcode the endpoint at the call sites. Before calling the API, resolve the base URL in this order:

1. If the user explicitly provides an endpoint in the current conversation, use it.
2. Otherwise, use the environment variable `OPENBIOMED_API_BASE_URL` if it is set in the runtime environment.
3. Otherwise, ask the user once which endpoint to use, and offer these options:
   - **OpenBioMed cloud service** (default, hosted): the `OPENBIOMED_CLOUD_URL` value declared above.
   - **Self-hosted OpenBioMed server**: the user provides their own base URL, e.g. `http://localhost:9000` or `https://openbiomed.internal.example.com`.
4. Remember the chosen base URL for the rest of the session and reuse it for subsequent calls without re-asking.

Privacy note: if the query data is proprietary or unpublished, recommend a self-hosted endpoint rather than the public cloud service, and let the user confirm before sending.

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL (no trailing slash). The full endpoint is `${OPENBIOMED_API_BASE_URL}/run_pipeline/`.

## When to Use

- User asks about innovative drugs for a specific disease
- User provides a disease name and wants drug discovery insights
- User needs comprehensive analysis of targets, mechanisms, and clinical progress
- User wants to understand R&D trends for a disease area

## API Query Types

The `disease_drug_intel` task supports the following query types via `query_type` parameter:

### ChEMBL Queries

| query_type | Description | Required Params | Optional Params |
|------------|-------------|-----------------|-----------------|
| `chembl_search_target` | Search targets in ChEMBL | `query` | `limit`, `offset` |
| `chembl_search_molecule` | Search molecules/drugs | `query` | `limit`, `offset` |
| `chembl_get_drug` | Get drug details | `chembl_id` | - |
| `chembl_get_molecule` | Get molecule details | `chembl_id` | - |
| `chembl_get_target` | Get target details | `chembl_id` | - |
| `chembl_get_mechanism` | Get mechanism of action | - | `molecule_chembl_id`, `limit` |
| `chembl_get_indication` | Get drug indications | - | `molecule_chembl_id`, `efo_term`, `limit` |

### ClinicalTrials Queries

| query_type | Description | Required Params | Optional Params |
|------------|-------------|-----------------|-----------------|
| `clinicaltrials_search` | Search clinical trials | - | `query_cond`, `query_term`, `filter_overall_status`, `fields`, `sort`, `page_size`, `count_total` |
| `clinicaltrials_get` | Get specific trial | `nct_id` | `fields` |

### Tavily Search

| query_type | Description | Required Params | Optional Params |
|------------|-------------|-----------------|-----------------|
| `search` | Web search via Tavily | `query` | `max_results`, `api_key` |

Note: Tavily search requires `langchain_tavily` package and `TAVILY_API_KEY` environment variable.

## Workflow

### Step 0: Task Structuring

Construct task object (example):
```json
{
  "task_type": "disease_to_drug",
  "focus": "innovative_drugs",
  "disease_raw": "diabetes",
  "time_constraint": null,
  "region_constraint": null,
  "stage_constraint": null
}
```

### Step 1: Disease Standardization

Output `canonical_disease`, `subtypes`, `aliases`, `preferred_query_terms`.
If the user does not specify a subtype, first perform overall disease analysis, then emphasize more active R&D subtypes (e.g., for diabetes, prioritize T2DM coverage).

### Step 2: Innovative Drug Definition Mapping

Map "innovative drugs" to executable criteria:
- New mechanism/new target (with first-in-class tendency)
- Recently approved representative drugs
- Late-stage pipeline candidates (Phase II/III priority)
- Frontier directions (dual/multi-target, next-generation optimized molecules)

### Step 3: Subtask Breakdown

Execute 5 fixed subtasks:
- `identify_targets_and_mechanisms`
- `retrieve_representative_drugs`
- `build_drug_profiles`
- `validate_clinical_progress`
- `summarize_trends`

### Step 4: Database Query Execution

#### 4.1 ChEMBL Target Search

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "disease_drug_intel", "query_type": "chembl_search_target", "query": "EGFR", "limit": 10}'
```

Response:
```json
{
  "task": "disease_drug_intel",
  "query_type": "chembl_search_target",
  "results": {
    "page_size": 10,
    "targets": [
      {"target_chembl_id": "CHEMBL203", "pref_name": "Epidermal growth factor receptor", ...}
    ]
  }
}
```

#### 4.2 ChEMBL Molecule/Drug Search

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "disease_drug_intel", "query_type": "chembl_search_molecule", "query": "osimertinib", "limit": 10}'
```

#### 4.3 ChEMBL Get Drug/Molecule Details

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "disease_drug_intel", "query_type": "chembl_get_drug", "chembl_id": "CHEMBL3545063"}'
```

#### 4.4 ChEMBL Get Mechanism of Action

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "disease_drug_intel", "query_type": "chembl_get_mechanism", "molecule_chembl_id": "CHEMBL3545063"}'
```

#### 4.5 ChEMBL Get Drug Indications

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "disease_drug_intel", "query_type": "chembl_get_indication", "molecule_chembl_id": "CHEMBL3545063"}'
```

#### 4.6 ClinicalTrials Search

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "disease_drug_intel", "query_type": "clinicaltrials_search", "query_cond": "lung cancer", "fields": ["NCTId", "BriefTitle", "OverallStatus"], "page_size": 20}'
```

Response:
```json
{
  "task": "disease_drug_intel",
  "query_type": "clinicaltrials_search",
  "results": {
    "studies": [
      {"protocolSection": {"identificationModule": {"nctId": "NCT000001"}}}
    ]
  }
}
```

#### 4.7 ClinicalTrials Get Specific Trial

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "disease_drug_intel", "query_type": "clinicaltrials_get", "nct_id": "NCT000001"}'
```

#### 4.8 Tavily Web Search (Optional)

```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "disease_drug_intel", "query_type": "search", "query": "latest EGFR inhibitor approval 2024", "max_results": 5}'
```

### Step 5: Evidence Integration and Deduplication

Primary key priority:
- Drug: `ChEMBL ID > Standard drug name > ClinicalTrials intervention`
- Target: `Gene symbol/Standard target name > Alias`

Must preserve alias and formulation information to avoid incorrect merging (e.g., different semaglutide formulations).

### Step 6: Innovation Screening and Ranking

Score 0-5 and rank comprehensively:
- `disease_relevance`
- `innovation`
- `clinical_maturity`
- `evidence_strength`
- `representativeness`

Output must be layered:
- Approved/validated representative innovative drugs
- Late-stage pipeline candidate drugs
- Frontier exploratory mechanism directions

### Step 7: Report Generation

Before generating the report, read the `## 10. Chinese Report Template (Standard Version)` in `references/disease_to_drug_playbook.md`.

By default, output the final Chinese report strictly following that template, not just free-form "including these contents". Chapter order, primary numbering, and main title skeleton must remain consistent:

- `{Disease Name} Innovative Drug Comprehensive Analysis Report`
- `1. Problem Overview`
- `2. Key Conclusions First`
- `3. Disease-Related Key Targets and Mechanisms`
- `4. Representative Innovative Drug List`
- `5. Clinical Trial Progress Overview`
- `6. R&D Trends and Judgments`
- `7. Results Notes and Limitations`

Only when the user explicitly requests "brief version/summary version/table version/specific format" may deviation from the standard template be allowed; if not explicitly requested, the standard template must be used.

### Step 8: Exception Handling

- Too many results: Select Top N (default 10) by representativeness + innovation.
- Too few results: Prioritize outputting target directions and neighboring mechanisms, not forcefully padding drug lists.
- Evidence conflicts: Clearly write "molecular evidence exists/clinical evidence limited".
- Missing constraints: Default `time_constraint=null, region_constraint=global`, and explicitly declare in report.

## Expected Outputs

| Query Type | Response Field | Output |
|------------|---------------|--------|
| `chembl_search_target` | `results` | Target search results with ChEMBL IDs |
| `chembl_search_molecule` | `results` | Molecule/drug search results |
| `chembl_get_drug` | `results` | Drug details by ChEMBL ID |
| `chembl_get_mechanism` | `results` | Mechanism of action data |
| `chembl_get_indication` | `results` | Drug indication data |
| `clinicaltrials_search` | `results` | Clinical trial search results |
| `clinicaltrials_get` | `results` | Specific trial details |
| `search` | `results` | Tavily search results |

## Error Handling

### Endpoint Unreachable

**Symptom**: curl returns "Connection refused" or timeout.

**Solution**: Verify the endpoint is reachable (`curl ${OPENBIOMED_API_BASE_URL}/healthz` should return "Service available"). If unreachable, re-resolve the base URL per the resolution order above.

### Tavily Search Not Available

**Symptom**: `search` query returns error about missing `langchain_tavily` or `TAVILY_API_KEY`.

**Solution**: Ensure `langchain_tavily` package is installed and `TAVILY_API_KEY` environment variable is set. If unavailable, skip web search step and note in report.

### ChEMBL/ClinicalTrials Rate Limiting

**Symptom**: API returns 429 status or empty results.

**Solution**: Reduce query frequency, increase `limit` parameter, or retry after a short delay.

### Invalid ChEMBL ID

**Symptom**: `chembl_get_*` queries return 404.

**Solution**: Verify the ChEMBL ID format (should be `CHEMBLxxxxxx`). Use `chembl_search_*` first to find valid IDs.

## Example Usage

**Input**: "Analyze innovative drugs for Alzheimer's disease"

**Step 4.1**: Search targets
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "disease_drug_intel", "query_type": "chembl_search_target", "query": "amyloid beta", "limit": 10}'
```

**Step 4.2**: Search molecules
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "disease_drug_intel", "query_type": "chembl_search_molecule", "query": "aducanumab", "limit": 5}'
```

**Step 4.6**: Search clinical trials
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "disease_drug_intel", "query_type": "clinicaltrials_search", "query_cond": "Alzheimer", "filter_overall_status": "RECRUITING", "page_size": 20}'
```

**Final Output**: Generate comprehensive Chinese report following the standard template.

## Quality Check Checklist

- Is disease standardization complete (including aliases/subtypes)?
- Is the mechanism-drug-clinical three-layer evidence chain provided?
- Is entity normalization and alias deduplication complete?
- Is the output layered (approved/late-stage/frontier)?
- Are limitations, conflicts, and uncertainties clearly stated?

## Reference Files

- [disease_to_drug_playbook.md](references/disease_to_drug_playbook.md): Complete SOP, routing strategy, internal JSON structures, Chinese report template.