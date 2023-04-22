##  Molecule-Text Retrieval

Molecule-text retrieval is a multi-modal task that aims to select the most relevant text description for a molecule (m2t) and the most relevant molecule a text describes (t2m).  

#### Features

- Supported models: SciBERT, KV-PLM, MoMu, BioMedGPT-1.6B and combination of models. 
- Supported dataset: PCdes.
- Supproted evaluation: MRR (Mean Reversed Rank), Recall@1, Recall@5, Recall@10.

*Warning*: We provide supervised learning and zero-shot evaluation, and the latter could only be applied to multi-modal models like MoMu and BioMedGPT-1.6B. 

#### Data Preparation

Install PCdes [here](https://github.com/thunlp/KV-PLM/tree/master/Ret) (`align_des_filt3.txt` and `align_smiles.txt`) and put them under `datasets/mtr/`. Install `pair.txt` [here](https://pan.baidu.com/s/1c1IDHiQ4df64rbgLaVFw9w) (`password is iv7a`) and put it under `datasets/mtr/momu_pretrain` (The file is used to filter out molecules appear in MoMu pretraining). 

#### Model Preparation

Install SciBERT from [Hugging Face](https://huggingface.co/allenai/scibert_scivocab_uncased) and put it under `ckpts/text_ckpts/`. You can also change the value of `"model_name_or_path"` to `"allenai/scibert_scivocab_uncased"` in the `config/` JSON file to download the PLM when running the code.

The multi-modal models are optional if you don't want to reproduce their results:

- Install MoMu checkpoints following instructions [here](https://github.com/ddz16/MoMu).
- Install KV-PLM checkpoints following instructions [here](https://github.com/thunlp/KV-PLM). You should also install `bpe_encoding.txt` and `bpe_vocab.txt` in the repository and put them under `assets/KV-PLM` if you want to experiment with KV-PLM*.
- Install BioMedGPT-1.6B checkpoint [here](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg) (`password is 7a6b`).

The above 3 checkpoints should be placed under `ckpts/fusion_ckpts/` .

#### Training and Evaluation

You can run scripts using bash under `scripts/mtr/`:

```bash
scripts/mtr
├── run.sh                      # evaluate MoMu/BioMedGPT-1.6B under supervised learning or zero-shot
├── run_baseline.sh             # train composed models (an arbitary molecule encoder and an arbitary text encoder)
├── train_kvplm.sh              # train KV-PLM
├── train_kvplm_star.sh         # train KV-PLM*
└── train_scibert.sh            # train SciBert
```

You can also modify the scripts or directly use the following command:

```bash
python tasks/mol_task/mtr.py \
--device DEVICE \                     # gpu device id
--mode MODE \                         # training mode, select from [train, zero_shot]
--config_path CONFIG_PATH \           # configuration file, see configs/mtr/ for more details
--dataset DATASET \                   # dataset name, now only PCdes is available
--dataset_path DATASET_PATH \         # path to the dataset
--dataset_mode DATASET_MODE \         # task mode, select from [paragraph, sentence]
--filter \                            # to filter out drugs from MoMu pre-trained data or not
--filter_path FILTER_PATH \           # path to the MoMu pre-trained drugs
--init_checkpoint INIT_CHECKPOINT \   # checkpoint path
--param_key PARAM_KEY \               # key of the checkpoint dict that contains model parameters
--output_path OUTPUT_PATH \           # save checkpoint path
--num_workers NUM_WORKERS \           # number of workers when loading data
--epochs EPOCHS \                     # number of training epochs
--warmup WARMUP \                     # ratio of warmup epochs
--patience PATIENCE \                 # number of tolerant epochs for early-stopping
--weight_decay WEIGHT_DECAY \         # weight decay, default is 1e-5
--lr LR \                             # learning rate, default is 1e-3
--train_batch_size TRAIN_BATCH_SIZE \ # batch size for training, default is 32
--val_batch_size VAL_BATCH_SIZE \     # batch size for validation and test, default is 64
--log_every LOG_EVERY \               # steps for printing training information
--seed SEED \                         # random seed
--margin MARGIN                       # margin value in contrastive loss
```

