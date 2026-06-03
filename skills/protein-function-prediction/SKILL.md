---
name: protein-function-prediction
description: >
  Call interface for the OpenBioMed protein function prediction service. Calls the /run_pipeline/ endpoint
  of any OpenBioMed-compatible HTTP service with task="protein_question_answering" and model="biot5".
  Endpoint is configurable so this skill works against the OpenBioMed cloud service, a user-hosted instance,
  or a local dev server, independent of the underlying server implementation.
  Use this skill when:
  (1) You have a protein sequence and want to understand its biological function,
  (2) You need to identify enzyme activity, pathway involvement, or molecular interactions,
  (3) You want a concise description of protein properties from sequence alone.
license: MIT
category: protein-engineering
tags: [protein, function-prediction, annotation, sequence-analysis, biot5]
---

# Protein Function Prediction

Predict functional annotations and properties for proteins from their amino acid sequences by calling the `/run_pipeline/` endpoint of an OpenBioMed service with `task="protein_question_answering"` and `model="biot5"`.

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

- You have a protein FASTA sequence and need to understand its biological role
- You want to identify enzyme function, pathway involvement, or molecular mechanisms
- You need quick functional insights without experimental data
- You're characterizing novel or unannotated protein sequences

## Workflow

You can query the /run_pipeline/ endpoint with a protein sequence and a related question to perform inference and obtain protein function predictions.
The request payload must be in JSON format. The required fields are protein (the protein sequence) and text (the question text).
Other optional parameters can be found in the TaskRequest definition. Any parameters not defined in TaskRequest are strictly prohibited in the request payload.


```json
{
  "task": "protein_question_answering",
  "model": "biot5",
  "protein": "YOUR_AMINO_ACID_SEQUENCE",
  "text": "user question about the protein sequence"
}


## API Call Example:
Replace YOUR_AMINO_ACID_SEQUENCE with the protein sequence to be analyzed, and user question about the protein sequence with your specific question.

curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "protein_question_answering",
  "model": "biot5",
  "protein": "YOUR_AMINO_ACID_SEQUENCE",
  "text": "user question about the protein sequence"
}'

## 常见使用场景

### 1. 酶活性预测
```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "protein_question_answering",
  "model": "biot5",
  "protein": "YOUR_AMINO_ACID_SEQUENCE",
  "text": "这种蛋白质是酶吗？如果是，它的催化活性和催化的反应是什么？"
}'
```

### 2. 通路分析
```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "protein_question_answering",
  "model": "biot5",
  "protein": "YOUR_AMINO_ACID_SEQUENCE",
  "text": "这种蛋白质参与什么生物通路？描述它的作用。"
}'
```

### 3. 蛋白质家族分类
```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "protein_question_answering",
  "model": "biot5",
  "protein": "YOUR_AMINO_ACID_SEQUENCE",
  "text": "这个序列属于哪个蛋白质家族？描述它的结构特征。"
}'
```

### 4. 跨膜结构域预测
```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "protein_question_answering",
  "model": "biot5",
  "protein": "YOUR_AMINO_ACID_SEQUENCE",
  "text": "这种蛋白质是否包含跨膜结构域？如果有，描述它的膜拓扑结构。"
}'
```

### 5. 蛋白互作预测
```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "protein_question_answering",
  "model": "biot5",
  "protein": "YOUR_AMINO_ACID_SEQUENCE",
  "text": "这种蛋白质可能参与哪些蛋白质相互作用？"
}'
```

### 6. 信号肽与亚细胞定位分析
```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "protein_question_answering",
  "model": "biot5",
  "protein": "YOUR_AMINO_ACID_SEQUENCE",
  "text": "这种蛋白质有信号肽吗？它的细胞定位是什么？"
}'
```

### 7. 药物靶点潜力评估
```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "protein_question_answering",
  "model": "biot5",
  "protein": "YOUR_AMINO_ACID_SEQUENCE",
  "text": "这种蛋白质是潜在的药物靶点吗？说明它的适用性。"
}'
```

### 8. 疾病关联分析
```bash
curl -X 'POST' \
  '${OPENBIOMED_API_BASE_URL}/run_pipeline/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task": "protein_question_answering",
  "model": "biot5",
  "protein": "YOUR_AMINO_ACID_SEQUENCE",
  "text": "这种蛋白质与人类疾病有关吗？描述任何已知的疾病关联。"
}'
```

## Limitations

- Sequences longer than 512 residues are truncated
- Model trained on known proteins; novel folds may have lower accuracy
- Does not predict 3D structure or binding sites (use `protein_folding` or `protein_binding_site_prediction` tools)

## Related Skills

- `protein-structure-design-boltzgen`: For 3D structure prediction
- `protein-mutation-analysis`: For mutation effect prediction
- `uniprot-query`: For retrieving protein metadata from UniProt

