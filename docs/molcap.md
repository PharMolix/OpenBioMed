##  Molecule Captioning

Molecule captioning is a multi-modal task that aims to generate texts for a molecule that describes its funcitons and properties. 

#### Features

- Supported models: [MolT5](https://arxiv.org/abs/2204.11817), [GraphMVP](https://arxiv.org/abs/2110.07728) with MolT5 decoder, [MoMu](https://arxiv.org/abs/2209.05481), BioMedGPT-1.6B, and [MolFM](https://arxiv.org/abs/2307.09484). 
- Supported dataset: ChEBI-20.
- Supported evaluation: BLEU-2, BLEU-4, ROUGE score, METEOR score, Text2Mol score.

#### Additional Packages

To evaluate the performance of molecule captioning, run the following:

```bash
pip install spacy
pip install rouge_score

pip install nltk
python
>>> import nltk
>>> nltk.download('wordnet')
>>> nltk.download('omw-1.4')
```

#### Data Preparation

Install [ChEBI-20](https://github.com/blender-nlp/MolT5/tree/main/ChEBI-20_data) and put the files under `datasets/molcap/chebi-20`. For Text2Mol evaluation, download `cids_to_smiles.pkl` [here](https://uofi.box.com/v/MolT5-cid-to-smiles) and `test.txt`  [here](https://github.com/blender-nlp/MolT5/tree/main/evaluation/text2mol_data). Put them under `assets/text2mol`.

#### Model Preparation
Install [SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased) and [MolT5](https://huggingface.co/laituan245) and put them under `ckpts/text_ckpts/`. Distinguish between MolT5-smiles2caption (fine-tuned for molecule caption) and MolT5 (not fine-tuned). 

To reproduce **MoMu**, install MoMu checkpoints following instructions [here](https://github.com/ddz16/MoMu) and put it under `ckpts/fusion_ckpts/momu`.

To reproduce **MolFM** and **BioMedGPT-1.6B**, install the checkpoint [here](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg) (`password is 7a6b`) and put them under `ckpts/fusion_ckpts/`.

For Text2Mol evaluation, download the Text2Mol checkpoint `test_outputfinal_weights.320.pt` [here](https://uofi.box.com/s/es16alnhzfy1hpagf55fu48k49f8n29x) and put it under `ckpts/fusion_ckpts/text2mol/`.

#### Training and Evaluation

You can run the Bash scripts under `scripts/multimodal/molcap/`:

```bash
scripts/multimodal/molcap/
├── train.sh										# train molecule captioning model
├── test.sh											# test molecule captioning model
└── test_from_file.sh						# report evaluation metrics for molecule captions within a given file
```

Example:

```bash
bash scripts/multimodal/molcap/train.sh cuda:0 # switch to your own cuda device or cpu
```

You can also modify the scripts or directly use the following command:

```bash
python open_biomed/tasks/multi_modal_task/molcap.py \
--device DEVICE \                         # gpu device id
--mode MODE \                             # traning mode, select from [train, test, traintest]
--config_path CONFIG_PATH \               # configuration file, see configs/molcap/ for more details
--dataset DATASET \                       # dataset name, now only PCdes is available
--dataset_path DATASET_PATH \             # path to the dataset
--output_path OUTPUT_PATH \               # path to save checkpoint for training
--caption_save_path CAPTION_SAVE_PATH \   # path to save generated captions
--num_workers NUM_WORKERS \               # number of workers when loading data
--epochs EPOCHS \                         # number of training epochs
--patience PATIENCE \                     # number of tolerant epochs for early-stopping
--weight_decay WEIGHT_DECAY \             # weight decay, default is 0
--lr LR \                                 # learning rate, default is 1e-4
--batch_size BATCH_SIZE \                 # batch size, default is 32
--logging_steps LOGGING_STEPS \           # steps for printing training information
--text2mol_bert_path TEXT2MOL_BERT_PATH \ # path to scibert
--text2mol_data_path TEXT2MOL_DATA_PATH \ # path to `cids_to_smiles.pkl`
--text2mol_ckpt_path TEXT2MOL_CKPT_PATH   # path to Text2Mol checkpoint

# If you follow the preparation in the above sections, you can simply set text2mol arguments to default values.
```

