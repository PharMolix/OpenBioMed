## Protein-protein Interaction Prediction

Protein-protein interaction prediction aims to identify and catalog physical interactions between pairs or groups of proteins. It is beneficial to gaining insights into various biochemical processes.

#### Features

- Supported models: [PIPR](https://academic.oup.com/bioinformatics/article/35/14/i305/5529260), [GNN-PPI](https://arxiv.org/abs/2105.06709) and [OntoProtein](https://arxiv.org/abs/2201.11147). This is a continuing effort and we are working on further growing the list.
- Supported datasets: SHS27k, SHS148k and STRING.
- Supported split: Random; DFS; BFS.
- Supported evaluation: Micro F1.

#### Data Preparation

All the datasets can downloaded [here](https://drive.google.com/file/d/12d5wzNcuPxPyW8KIzwmvGg2dOKo0K0ag/view), and should put under `datasets/ppi/`. 

#### Model Preparation

To reproduce **OntoProtein**, download the model from [HuggingFace](https://huggingface.co/zjunlp/OntoProtein) and put it under `ckpts/protein_ckpts/`.

#### Training and Evaluation

You can run scripts using bash under `scripts/dti/`:

```bash
scripts/aidd/ppi
├── run.sh             # running models
```

Example:

```bash
bash scripts/aidd/ppi cuda:0 # switch to your own cuda device or cpu
```

You can also modify the scripts or directly use the following command:

```bash
python open_biomed/tasks/prot_task/ppi.py \
--device DEVICE \                   # gpu device id
--mode MODE \                       # training mode, train / test
--config_path CONFIG_PATH \         # configuration file, see configs/ppi/ for more details
--dataset DATASET \                 # dataset name, select from [SHS27k, SHS148k, STRING]
--dataset_path DATASET_PATH \       # path to the dataset
--split_strategy SPLIT_STRATEGY \   # split strategy, select from [random, bfs, dfs]
--init_checkpoint INIT_CHECKPOINT \ # checkpoint path used for efficient validation
--param_key PARAM_KEY \             # key of the checkpoint dict that contains model parameters
--output_path OUTPUT_PATH \         # save checkpoint path used for training
--num_workers NUM_WORKERS \         # number of workers when loading data
--epochs EPOCHS \                   # number of training epochs
--patience PATIENCE \               # number of tolerant epochs for early-stopping
--weight_decay WEIGHT_DECAY \       # weight decay, default is 1e-5
--lr LR \                           # learning rate, default is 1e-3
--batch_size BATCH_SIZE \           # batch size, default is 512
--logging_steps LOGGING_STEPS \     # steps for printing training information
--seed SEED                         # random seed
```

