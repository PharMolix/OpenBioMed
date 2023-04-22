##  Cell Type Classification

#### Features

- Supported models: ScBERT and CellLM. 
- Supported dataset: Zheng68k.
- Supported evaluation: Accuracy and F1 Score.

#### Additional Packages

To support usage of cell type classification, run the following commands:
```
pip install scanpy
pip install einops
```

#### Data Preparation

Download Zheng68k [here](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg) (password is `7a6b`) and put `BioMedGPT/data/zheng68k_hgnc.h5ad` under `OpenBioMed/datasets/ctc/zheng68k/`.

#### Model Preparation

Install ScBERT following instructions [here](https://github.com/TencentAILabHealthcare/scBERT) and CellLM [here]( https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg) (password is `7a6b`) and put them under `ckpts/cell_ckpts/`. 

#### Training and Evaluation

You can run the Bash script `open_biomed/scripts/ctc/train.sh`:

You can also modify the script or directly use the following command:

```bash
python tasks/mol_task/ctc.py \
--device DEVICE \                         # gpu device id
--mode MODE \                             # traning mode, select from [train, zero_shot]
--config_path CONFIG_PATH \               # configuration file, see configs/mtr/ for more details
--dataset DATASET \                       # dataset name, now only PCdes is available
--dataset_path DATASET_PATH \             # path to the dataset
--output_path OUTPUT_PATH \               # save checkpoint path
--epochs EPOCHS \                         # number of training epochs
--patience PATIENCE \                     # number of tolerant epochs for early-stopping
--lr LR \                                 # learning rate, default is 1e-4
--batch_size BATCH_SIZE \                 # batch size for training, default is 3
--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS # steps for gradient accumulation
--logging_steps LOGGING_STEPS \           # steps for printing training information
--seed SEED \                             # random seed
--unassign_threshold UNASSIGN_THRESHOLD   # threshold below which the predicted value is not assigned  a label
```

