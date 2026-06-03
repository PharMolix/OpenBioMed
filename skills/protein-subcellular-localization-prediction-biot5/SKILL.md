---
name: protein-subcellular-localization-prediction-biot5
description: >
  Call interface for protein subcellular localization prediction via the /run_pipeline/ endpoint of any OpenBioMed-compatible HTTP service.
  Endpoint is configurable so this skill works against the OpenBioMed cloud service, a user-hosted instance, or a local dev server,
  independent of the underlying server implementation.
  Use this skill when:
  (1) You have a protein sequence and want to know where it localizes in the cell,
  (2) You need to identify cellular compartment (nucleus, cytoplasm, membrane, etc.),
  (3) You want quick localization prediction without experimental data.
license: MIT
category: protein-engineering
tags: [protein, subcellular-localization, localization, sequence-analysis, biot5]
---

# Protein Subcellular Localization Prediction Call Interface

Predict subcellular localization for proteins from their amino acid sequences via /run_pipeline/ interface using the BioT5 model.

## Endpoint Configuration (read this first)

Defaults declared in this skill (edit these inline when the real values are known):

- `OPENBIOMED_CLOUD_URL = http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520`
  Placeholder for the OpenBioMed cloud service base URL. Replace with the real published URL when available.

This skill does NOT hardcode the endpoint at the call sites. Before calling the API, resolve the base URL in this order:

1. If the user explicitly provides an endpoint in the current conversation, use it.
2. Otherwise, use the environment variable `OPENBIOMED_API_BASE_URL` if it is set in the runtime environment.
3. Otherwise, ask the user once which endpoint to use, and offer these options:
   - **OpenBioMed cloud service** (default, hosted): the `OPENBIOMED_CLOUD_URL` value declared above.
   - **Self-hosted OpenBioMed server**: the user provides their own base URL, e.g. `http://lb-2na6qnsx-c6103exlpimzja5q.clb.sh-tencentclb.net:32520` or `https://openbiomed.internal.example.com`.
4. Remember the chosen base URL for the rest of the session and reuse it for subsequent calls without re-asking.

Privacy note: if the protein sequence is proprietary or unpublished, recommend a self-hosted endpoint rather than the public cloud service, and let the user confirm before sending.

In the rest of this document, `${OPENBIOMED_API_BASE_URL}` is a placeholder for the resolved base URL (no trailing slash). The full endpoint is therefore `${OPENBIOMED_API_BASE_URL}/run_pipeline/`.

## When to Use

- You have a protein FASTA sequence and need to know its cellular location
- You want to identify if a protein is cytoplasmic, nuclear, membrane-bound, or secreted
- You need quick localization insights without experimental data
- You're characterizing novel or unannotated protein sequences

## API Parameters

**Required parameters:**
- `task`: "protein_question_answering"
- `model`: "biot5" (recommended)
- `protein`: Protein sequence in FASTA format
- `text`: Question about subcellular localization

```json
{
  "task": "protein_question_answering",
  "model": "biot5",
  "protein": "YOUR_AMINO_ACID_SEQUENCE",
  "text": "What is the subcellular localization of this protein?"
}
```

## API Call Examples

### 1. Basic Subcellular Localization Prediction

```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "protein_question_answering",
  "model": "biot5",
  "protein": "YOUR_AMINO_ACID_SEQUENCE",
  "text": "What is the subcellular localization of this protein?"
}'
```

### 2. Detailed Localization Analysis

```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "protein_question_answering",
  "model": "biot5",
  "protein": "YOUR_AMINO_ACID_SEQUENCE",
  "text": "Describe the subcellular localization and any membrane association or signal peptides."
}'
```

### 3. Secreted Protein Detection

```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "protein_question_answering",
  "model": "biot5",
  "protein": "YOUR_AMINO_ACID_SEQUENCE",
  "text": "Is this protein secreted? Does it have a signal peptide?"
}'
```

### 4. Nuclear Protein Identification

```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "protein_question_answering",
  "model": "biot5",
  "protein": "YOUR_AMINO_ACID_SEQUENCE",
  "text": "Is this protein localized to the nucleus? Does it contain nuclear localization signals?"
}'
```

### 5. Membrane Protein Analysis

```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "protein_question_answering",
  "model": "biot5",
  "protein": "YOUR_AMINO_ACID_SEQUENCE",
  "text": "Is this a membrane protein? Describe its membrane topology and any transmembrane domains."
}'
```

### 6. Mitochondrial Localization

```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "protein_question_answering",
  "model": "biot5",
  "protein": "YOUR_AMINO_ACID_SEQUENCE",
  "text": "Is this protein localized to mitochondria? Does it have mitochondrial targeting signals?"
}'
```

## 常见使用场景

### 1. 基础亚细胞定位预测

```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "protein_question_answering",
  "model": "biot5",
  "protein": "YOUR_AMINO_ACID_SEQUENCE",
  "text": "这种蛋白质的亚细胞定位是什么？"
}'
```

### 2. 信号肽与分泌蛋白检测

```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "protein_question_answering",
  "model": "biot5",
  "protein": "YOUR_AMINO_ACID_SEQUENCE",
  "text": "这种蛋白质是分泌蛋白吗？是否有信号肽？"
}'
```

### 3. 膜蛋白分析

```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "protein_question_answering",
  "model": "biot5",
  "protein": "YOUR_AMINO_ACID_SEQUENCE",
  "text": "这是一种膜蛋白吗？描述它的膜拓扑结构。"
}'
```

### 4. 线粒体定位

```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "protein_question_answering",
  "model": "biot5",
  "protein": "YOUR_AMINO_ACID_SEQUENCE",
  "text": "这种蛋白质是否定位在线粒体？是否有线粒体靶向信号？"
}'
```

## Limitations

- Sequences longer than 512 residues are truncated
- Predictions are based on sequence patterns; experimental validation recommended
- Novel protein families may have lower prediction accuracy

## Related Skills

- `protein-function-prediction`: For comprehensive function prediction
- `protein-structure-design-boltzgen`: For 3D structure prediction
- `protein-mutation-analysis`: For mutation effect analysis
- `uniprot-query`: For retrieving protein metadata from UniProt
