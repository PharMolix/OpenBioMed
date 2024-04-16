##  Molecule-Text Retrieval

Molecule-text retrieval is a multi-modal task that aims to select the most relevant text description for a molecule (m2t) and the most relevant molecule a text describes (t2m).  

#### Features

- Supported models: [SciBERT](https://arxiv.org/abs/1903.10676), [KV-PLM](https://www.nature.com/articles/s41467-022-28494-3), [MoMu](https://arxiv.org/abs/2209.05481), BioMedGPT-1.6B, [GraphMVP](https://arxiv.org/abs/2110.07728), and [MolFM](https://arxiv.org/abs/2307.09484). 
- Supported dataset: PCdes.
- Supproted evaluation: MRR (Mean Reversed Rank), Recall@1, Recall@5, Recall@10.

*Warning*: We provide supervised setting and zero-shot setting. The latter should only be applied to pre-trained multimodal molecular models like MoMu, MolFM and BioMedGPT-1.6B. 

#### Data Preparation

Install [PCdes](https://github.com/thunlp/KV-PLM/tree/master/Ret) (`align_des_filt3.txt` and `align_smiles.txt`) and put them under `datasets/mtr/`. Install `pair.txt` [here](https://pan.baidu.com/s/1c1IDHiQ4df64rbgLaVFw9w) (`password is iv7a`) and put it under `datasets/mtr/momu_pretrain` (The file is used to filter out molecules that overlap with pre-training data). 

#### Model Preparation

Install SciBERT from [Hugging Face](https://huggingface.co/allenai/scibert_scivocab_uncased) and put it under `ckpts/text_ckpts/`. 

To reproduce **MoMu**, install checkpoints following instructions [here](https://github.com/ddz16/MoMu) and put it under `ckpts/fusion_ckpts/momu`.

To reproduce **KV-PLM**, install checkpoints following instructions [here](https://github.com/thunlp/KV-PLM) and put it under `ckpts/text_ckpts/kvplm`. You should also install `bpe_encoding.txt` and `bpe_vocab.txt` in the repository and put them under `assets/KV-PLM` if you want to experiment with KV-PLM*.

To reproduce **MolFM** and **BioMedGPT-1.6B**, install checkpoints [here](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg) (`password is 7a6b`).

The above 3 checkpoints should be placed under `ckpts/fusion_ckpts/` .

#### Training and Evaluation

You can run scripts using bash under `scripts/multimodal/mtr/`:

```bash
scripts/multimodal/mtr
├── run_multimodal.sh           # train MoMu/BioMedGPT-1.6B/MolFM under supervised or zero-shot settings
└── run_baseline.sh             # train composed models (an arbitary molecule encoder and an arbitary text encoder)
```

Example:
```bash
bash scripts/multimodal/mtr/run_multimodal.sh cuda:0 # switch to your own cuda device or cpu
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

