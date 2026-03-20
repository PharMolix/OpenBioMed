# Local Usage

## Repo Root

`/DATA/disk0/zhaosy/home/LangCell`

## Environment

The official `requirements.txt` pins:

- `torch==2.0.1`
- `geneformer==0.0.1`
- `transformers==4.39.1`
- `datasets==2.14.0`
- `scanpy==1.9.3`

Python in the README is `>3.9.18`.

## Required External Assets

The repo does not ship the actual pretrained checkpoints or ontology/json assets.
The workflow typically needs:

- checkpoint modules for `cell_bert`, `text_bert`, `cell_proj.bin`, `text_proj.bin`, `ctm_head.bin`
- tokenized dataset saved with `datasets.save_to_disk(...)`
- ontology or text-description JSON such as `obo.json` / `type2text.json`

If these files are missing, stop and ask for their locations instead of guessing.

## Workflow Selection

| Goal | Best entry point |
|---|---|
| first LangCell demo | `LangCell-annotation-zeroshot/zero-shot.ipynb` |
| tokenizing a new AnnData dataset | `data_preprocess/preprocess.py` |
| multimodal few-shot cell typing | `LangCell-annotation-fewshot/fewshot.py` |
| cell-encoder-only supervised training | `LangCell-CE-annotation/finetune.py` |
| cell-encoder-only few-shot | `LangCell-CE-annotation/fewshot.py` |

## Preprocessing Pattern

Minimal pattern from `data_preprocess/preprocess.py`:

```python
from utils import LangCellTranscriptomeTokenizer
import scanpy as sc

data = sc.read_h5ad("/path/to/adata.h5ad")
data.obs["n_counts"] = data.X.sum(axis=1)
data.var["ensembl_id"] = data.var["feature_id"]

tk = LangCellTranscriptomeTokenizer(dict([(k, k) for k in data.obs.keys()]), nproc=4)
tokenized_cells, cell_metadata = tk.tokenize_anndata(data)
tokenized_dataset = tk.create_dataset(tokenized_cells, cell_metadata)
tokenized_dataset.save_to_disk("/path/to/tokenized_dataset")
```

## Zero-Shot Logic

The zero-shot notebook does four important things:

1. loads `cell_bert`, `text_bert`, projection heads, and `ctm_head`
2. loads a tokenized dataset from disk
3. maps candidate cell types to textual descriptions
4. combines:
   - cell-text embedding similarity
   - cell-text matching scores from the multimodal text encoder

Final prediction is based on the average of the two score matrices by default.

## Few-Shot Logic

`LangCell-annotation-fewshot/fewshot.py`:

- loads a tokenized dataset from disk
- normalizes label column names to `celltype`
- builds text prompts from ontology descriptions
- samples `nshot` labeled cells per class
- trains with both similarity loss and CTM loss

## LangCell-CE Finetuning Logic

`LangCell-CE-annotation/finetune.py`:

- loads a tokenized dataset from disk
- converts string labels to integer ids
- splits train/eval after shuffling
- loads `BertForSequenceClassification` from the cell encoder path
- patches the embedding layer if vocab size differs from the tokenizer dictionary
- trains with Hugging Face `Trainer`

## Practical Checks

Before running LangCell, verify:

1. the dataset is already tokenized and readable by `load_from_disk`
2. the label column names match one of the names checked by the repo
3. the candidate text descriptions are biologically meaningful
4. checkpoint module paths are all present
5. GPU device and `CUDA_VISIBLE_DEVICES` settings are correct
