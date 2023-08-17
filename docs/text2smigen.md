##  Text-to-SMILES Generation

Text-to-SMILES generations, also called text-based de novo molecule generation, is a task that aims to generate SMILES strings based on text descriptions.  

#### Features

- Supported models: [MolT5](https://arxiv.org/abs/2204.11817), [MoMu](https://arxiv.org/abs/2209.05481), BioMedGPT-1.6B, and [MolFM](https://arxiv.org/abs/2307.09484). 
- Supported dataset: ChEBI-20.
- Supproted evaluation: BLEU, Exact ratio, Valid ratio, Levenshtein distance, MACCS fingerprint similarity, RDKit fingerprint similarity, Morgan fingerprint similarity and Text2Mol score.

Pipelines that generate SMILES without calculating evaluation metrics will be developed in the future.

#### Addtional Packages

To evaluate the performance of molecule captioning, run the following:

```bash
pip install Levenshtein

pip install nltk
python
>>> import nltk
>>> nltk.download('wordnet')
>>> nltk.download('omw-1.4')
```

To evaluate models with Text2Mol metric, install [mol2vec](https://pubs.acs.org/doi/10.1021/acs.jcim.7b00616) using the following command:
```bash
cd assets
git clone https://github.com/samoturk/mol2vec
cd mol2vec
pip install .
```

Install the mol2vec checkpoint [here](https://github.com/blender-nlp/MolT5/blob/main/evaluation/m2v_model.pkl) and put it under `assets`.

#### Data Preparation

Follow `Data Preparation` in [molecule captioning](./molcap.md) to install the ChEBI-20 dataset and Text2Mol model.

#### Model preparation
Install [SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased) and [MolT5](https://huggingface.co/laituan245) and put them under `ckpts/text_ckpts/`. Distinguish between MolT5-caption2smiles (fine-tuned for text-to-SMILES generation) and MolT5 (not fine-tuned).

To reproduce **MoMu**, install checkpoints following instructions [here](https://github.com/ddz16/MoMu) and put it under `ckpts/fusion_ckpts/momu`.

To reproduce **MolFM** and **BioMedGPT-1.6B**, install checkpoints [here](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg) (`password is 7a6b`) and put them under `ckpts/fusion_ckpts/` .

#### Training and Evaluation

You can run the Bash scripts under `scripts/multimodal/text2smi/`:

```bash
scripts/multimodal/text2smi/
├── evaluate_text2mol.sh         # calculate Text2Mol score with generated results from file
├── test_molt5.sh                # test MolT5 model
└── train.sh                     # train SciBERT / MoMu / MolFM / BioMedGPT-1.6B
```

Example:

```bash
bash scripts/multimodal/text2smi/train.sh cuda:0 # switch to your own cuda device or cpu
```

You can also modify the scripts or directly use the following command:

```bash
python open_biomed/tasks/multimodal_task/text2smigen.py \
--device DEVICE \                         # gpu device id
--mode MODE \                             # traning mode, select from [train, test, traintest, test_text2mol]
--config_path CONFIG_PATH \               # configuration file, see configs/text2smigen/ for more details
--dataset DATASET \                       # dataset name, now only PCdes is available
--dataset_path DATASET_PATH \             # path to the dataset
--smi_save_path SMI_SAVE_PATH \           # path to save the generated SMILES
--output_path OUTPUT_PATH \               # checkpoint save path
--num_workers NUM_WORKERS \               # number of workers when loading data
--epochs EPOCHS \                         # number of training epochs
--patience PATIENCE \                     # number of tolerant epochs for early-stopping
--weight_decay WEIGHT_DECAY \             # weight decay, default is 0
--lr LR \                                 # learning rate, default is 1e-4
--batch_size TRAIN_BATCH_SIZE \           # batch size for training, default is 32
--logging_steps LOGGING_STEPS \           # steps for printing training information
--text2mol_bert_path TEXT2MOL_BERT_PATH \ # path to scibert
--text2mol_data_path TEXT2MOL_DATA_PATH \ # path to `cids_to_smiles.pkl`
--text2mol_ckpt_path TEXT2MOL_CKPT_PATH   # path to Text2Mol checkpoint
```

For calculating Text2Mol score, please refer to `scripts/text2smi/evaluate_text2mol.sh`.