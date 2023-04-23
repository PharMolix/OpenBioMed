##  Text-to-SMILES Generation

Text-to-SMILES generations, also called text-based de novo molecule generation, is a task that aims to generate SMILES strings based on text descriptions.  

#### Features

- Supported models: MolT5, MoMu, and BioMedGPT-1.6B. 
- Supported dataset: ChEBI-20.
- Supproted evaluation: BLEU, Exact ratio, Valid ratio, Levenshtein distance, MACCS fingerprint similarity, RDKit fingerprint similarity, Morgan fingerprint similarity and Text2Mol score.

Pipelines that generate SMILES without calculating evaluation metrics will be developed in the future.

#### Addtional Packages

To evaluate the performance of molecule captioning, run the following:

```bash
# NOTE: make sure you are at BioMed/ directory
pip install Levenshtein

pip install nltk
python
>>> import nltk
>>> nltk.download('wordnet')
>>> nltk.download('omw-1.4')

# mol2vec is required if you want to evaluate models with Text2Mol metric
cd assets
git clone https://github.com/samoturk/mol2vec
cd mol2vec
pip install .
```

#### Data Preparation

Follow `Data Preparation` in [molcap.md](./mocap.md) to install the ChEBI-20 dataset and Text2Mol model.

#### Model preparation
Install [SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased) and [MolT5](https://huggingface.co/laituan245) and put them under `ckpts/text_ckpts/`. Distinguish between MolT5-caption2smiles (fine-tuned for text-to-SMILES generation) and MolT5 (not fine-tuned). You can also change the value of `"model_name_or_path"` to `"allenai/scibert_scivocab_uncased"` or `"laituan245/molt5-[small/base/large]"`in the config json file to download the PLM when running the code.

The multi-modal models are optional if you don't want to reproduce their results:

- Install MoMu checkpoints following instructions [here](https://github.com/ddz16/MoMu).
- Install BioMedGPT-1.6B checkpoint [here](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg) (`password is 7a6b`).

The above 2 checkpoints should be placed under `ckpts/fusion_ckpts/` .

#### Training and Evaluation

You can run the Bash scripts under `scripts/text2smi/`:

```bash
scripts/text2smi/
├── evaluate_text2mol.sh         # calculate Text2Mol score with generated results from file
├── test_molt5.sh                # test MolT5 model
└── train.sh                     # train MoMu / BioMedGPT-1.6B with MolT5 as decoder
```

You can also modify the scripts or directly use the following command:

```bash
python tasks/mol_task/mtr.py \
--device DEVICE \                         # gpu device id
--mode MODE \                             # traning mode, select from [train, test, traintest, test_text2mol]
--config_path CONFIG_PATH \               # configuration file, see configs/mtr/ for more details
--dataset DATASET \                       # dataset name, now only PCdes is available
--dataset_path DATASET_PATH \             # path to the dataset
--smi_save_path SMI_SAVE_PATH \           # path to save the generated SMILES
--output_path OUTPUT_PATH \               # save checkpoint path
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