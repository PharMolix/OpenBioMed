---
name: target-drug-report
description: |
  Generate comprehensive drug development progress reports for disease therapeutic targets.
  Use when user asks about target drug pipeline, clinical trials, or research progress.
  Triggers on phrases like "target report", "drug development progress", "clinical trial summary",
  "靶点报告", "药物研发进展", "竞品分析", "专利分析".
license: MIT
category: drug-discovery
tags: [target-analysis, drug-pipeline, clinical-trials, market-analysis]
---

# Target Drug Development Report

Generate comprehensive, beautifully formatted reports on drug development progress for therapeutic targets via the run_pipeline API.

## Endpoint Configuration (read this first)

Defaults declared in this skill:

- `OPENBIOMED_CLOUD_URL = http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520`
  Placeholder for the OpenBioMed cloud service base URL.

This skill does NOT hardcode the endpoint at the call sites. Before calling the API, resolve the base URL in this order:

1. If the user explicitly provides an endpoint in the current conversation, use it.
2. Otherwise, use the environment variable `OPENBIOMED_API_BASE_URL` if it is set.
3. Otherwise, ask the user once which endpoint to use, offering these options:
   - **OpenBioMed cloud service** (default, hosted): the `OPENBIOMED_CLOUD_URL` value.
   - **Self-hosted OpenBioMed server**: user provides their own base URL.

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL (no trailing slash). The full endpoint is `${OPENBIOMED_API_BASE_URL}/run_pipeline/`.

## Inputs

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `target_name` | str | Yes | Target name (e.g., "CGRP", "EGFR", "KRAS") |
| `output_format` | str | No | "html" (default) or "markdown" |

## Workflow Overview

### Phase 1: Data Collection

| Step | API Call | Purpose |
|------|----------|---------|
| 1.1 | `web_search` | Search for target clinical trials |
| 1.2 | `literature_search` | Search PubMed for target research |
| 1.3 | `disease_drug_intel` | Get disease-drug intelligence |

### Phase 2: Report Generation

Local processing with collected data and built-in knowledge base.

---

## API Query Types

### Phase 1 APIs

#### web_search (Clinical Trials)
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "web_search", "query": "{target_name} clinical trial 2024 2025"}'
```

Response:
```json
{
  "task": "web_search",
  "text": "Search results containing clinical trial information..."
}
```

#### web_search (Research Papers)
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "web_search", "query": "{target_name} discovery mechanism site:pubmed.ncbi.nlm.nih.gov"}'
```

#### literature_search (PubMed)
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "literature_search", "query_type": "pubmed_search", "query": "{target_name} inhibitor", "max_results": 10}'
```

Response:
```json
{
  "task": "literature_search",
  "query_type": "pubmed_search",
  "results": [
    {"pmid": "12345678", "title": "...", "abstract": "...", "authors": [...]}
  ]
}
```

#### disease_drug_intel (ChEMBL Target Search)
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "disease_drug_intel", "query_type": "chembl_search_target", "target_name": "{target_name}"}'
```

Response:
```json
{
  "task": "disease_drug_intel",
  "query_type": "chembl_search_target",
  "results": [
    {"target_chembl_id": "CHEMBL...", "target_name": "..."}
  ]
}
```

#### disease_drug_intel (Clinical Trials Search)
```bash
curl -X POST "${OPENBIOMED_API_BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d '{"task": "disease_drug_intel", "query_type": "clinicaltrials_search", "query_term": "{target_name}"}'
```

Response:
```json
{
  "task": "disease_drug_intel",
  "query_type": "clinicaltrials_search",
  "results": [
    {"nct_id": "...", "title": "...", "phase": "..."}
  ]
}
```

Available query_types for disease_drug_intel:
- `chembl_search_target` - Search targets in ChEMBL
- `chembl_search_molecule` - Search molecules/drugs in ChEMBL
- `chembl_get_target` - Get target details by ChEMBL ID
- `chembl_get_mechanism` - Get mechanism of action for a molecule
- `chembl_get_indication` - Get drug indications
- `clinicaltrials_search` - Search clinical trials
- `clinicaltrials_get` - Get a specific clinical trial
- `search` - Web search via Tavily

---

## Complete Workflow Script

```bash
# Configuration
TARGET_NAME="CGRP"  # Replace with user's target
BASE_URL="${OPENBIOMED_API_BASE_URL}"

# Phase 1: Data Collection
echo "[Phase 1] Collecting target information..."

# Search clinical trials
CLINICAL_RESULT=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"web_search\", \"query\": \"${TARGET_NAME} clinical trial 2024 2025\"}")

# Search research papers
PAPER_RESULT=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"literature_search\", \"query_type\": \"pubmed_search\", \"query\": \"${TARGET_NAME} mechanism\", \"max_results\": 10}")

# Get disease-drug intelligence (ChEMBL target search)
DISEASE_RESULT=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"disease_drug_intel\", \"query_type\": \"chembl_search_target\", \"target_name\": \"${TARGET_NAME}\"}")

# Get clinical trials
CLINICALTRIALS_RESULT=$(curl -s -X POST "${BASE_URL}/run_pipeline/" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"disease_drug_intel\", \"query_type\": \"clinicaltrials_search\", \"query_term\": \"${TARGET_NAME}\"}")

# Phase 2: Report Generation
echo "[Phase 2] Generating report..."

# Combine API results with built-in knowledge base
# Report generation is done locally using the Python script:
# python skills/target-drug-report/examples/basic_example.py ${TARGET_NAME}

echo "Report generation complete!"
```

---

## Report Sections

| Section | Description | Data Source |
|---------|-------------|-------------|
| 🎯 靶点概况 | Target overview | Built-in KB + UniProt |
| 💊 已上市药物 | Approved drugs | Built-in KB + ChEMBL |
| 🏥 临床管线 | Clinical pipeline | web_search + disease_drug_intel |
| 📚 研究热点 | Research trends | literature_search |
| 📜 专利布局 | Patent landscape | web_search |
| 📊 市场分析 | Market analysis | Built-in KB |
| 🔮 投资展望 | Investment outlook | Analysis |

---

## Output Formats

### HTML (推荐)
- Modern responsive design
- Interactive charts and progress bars
- Color-coded status tags
- Print to PDF support

### Markdown
- Plain text format
- Table-based layout
- Version control friendly
- Easy to edit

---

## Built-in Knowledge Base

When web search is unavailable, uses built-in knowledge for common targets:

| Category | Targets |
|----------|---------|
| Oncology | EGFR, KRAS, BCL-2, ALK, BRAF, HER2 |
| Immunology | PD-1, CTLA-4 |
| Neurology | CGRP |
| Other | JAK, BTK |

---

## Expected Outputs

| Output | Description |
|--------|-------------|
| HTML file | `{target}_target_drug_report.html` |
| Markdown file | `{target}_target_drug_report.md` |
| Key statistics | Approved drugs, pipeline count, market size |

---

## Error Handling

### Web Search Unavailable

**Symptom**: API returns error or timeout.

**Solution**: Use built-in knowledge base for common targets.

### No Results for Target

**Symptom**: No clinical trial or research data found.

**Solution**: Target may be novel or less studied. Use broader search terms.

### Disease-Drug Intel Failed

**Symptom**: `disease_drug_intel` returns empty.

**Solution**: Target may not be mapped in database. Use web search instead.

---

## Limitations

- Built-in knowledge for ~20 common targets only
- Patent data may be incomplete
- Market analysis uses static estimates
- Requires network for real-time data

## Example Usage

**Input**: "Generate drug report for CGRP target"

**Workflow**:
1. Search CGRP clinical trials
2. Search CGRP PubMed papers
3. Get disease-drug intelligence
4. Generate HTML report with built-in data

**Input**: "EGFR 靶点药物研发进展"

**Workflow**:
1. Search EGFR clinical trials in Chinese
2. Get disease-drug intelligence for NSCLC
3. Generate bilingual report

---

## Related Skills

- `drug-candidate-discovery` - Generate drug candidates for target
- `disease-drug-intelligence` - Query disease-drug relationships
- `chembl-query` - Query ChEMBL for target drugs