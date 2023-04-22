##  Drug-Response Prediction

Drug-response prediction aims at predicting if a drug is responsive on cells, tissues, or patients, based on the gene expression profile.  

#### Features

- Supported models: TGDRP. More models will be supported in the future. 
- Supported dataset: GDSC, TCGA.
- Supproted evaluation: ROC_AUC for classification and RMSE, MAE, $r^2$ index and pearson coefficient.

#### Data Preparation

Install GDSC and TCGA [here](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg) (`password is 7a6b`, `GDSC/ and TCGA/`) and put them under `datasets/drp/`. Install supplementary data [here](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg) (`password is 7a6b`, `drp/`) and put them under `assets/drp/`. Install the STRING database (homo sapiens) [here](https://string-db.org/cgi/download) (`9606.protein.info.v11.0.txt`, `9606.protein.links.v11.0.txt` and `9606.protein.sequences.v11.0.fasta`) and put it under `assets/kg/STRING/`.

#### Training and Evaluation

You can run the Bash scripts under `dair_biomed/scripts/mtr/`:

```bash
open_biomed/scripts/drp/
├── train.sh				# train TGDRP on GDSC dataset
└── transfer.sh             # train TGDRP on GDSC dataset and zero-shot test on TCGA dataset
```

You can also modify the scripts or directly use the following command:

```bash
python tasks/mol_task/drp.py \
--device DEVICE \                   # gpu device id
--task TASK \                       # task type, select from [classification, regression]
--mode MODE \                       # training mode, select from [train, test, zero_shot_transfer]
--config_path CONFIG_PATH \         # configuration file, see configs/dti/ for more details
--dataset DATASET \                 # dataset name
--dataset_path DATASET_PATH \       # path to the dataset
--init_checkpoint INIT_CHECKPOINT \ # checkpoint path used for efficient validation
--param_key PARAM_KEY \             # key of the checkpoint dict that contains model parameters
--num_workers NUM_WORKERS \         # number of workers when loading data
--epochs EPOCHS \                   # number of training epochs
--patience PATIENCE \               # number of tolerant epochs for early-stopping
--weight_decay WEIGHT_DECAY \       # weight decay, default is 1e-4
--lr LR \                           # learning rate, default is 0
--batch_size BATCH_SIZE \           # batch size, default is 128
--transfer_dataset_path TRANSFER_DATASET_PATH \ # path to the zero-shot transfer dataset
--seed SEED           							# random seed
```

