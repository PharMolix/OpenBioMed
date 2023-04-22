## Drug Property Prediction

Drug property prediction (DP) aims to predict molecule properties such as toxicity and side effects, which is a significant task in drug discovery.  

#### Features

- Supported models: [MolCLR](https://github.com/yuyangw/MolCLR), [MGraphDTA](https://github.com/guaguabujianle/MGraphDTA), [MoMu](https://github.com/ddz16/MoMu) and **DeepEIK**. More models will be implemented and more combinations will be tested in the future.
- Supported datasets: 8 classification datasets i.e. BBBP, Tox21, ToxCast, SIDER, ClinTox, MUV, HIV and BACE of [MoleculeNet](https://moleculenet.org).
- Supported split: random split, scaffold split and random-scaffold split;
- Supproted evaluation: ROC_AUC.

#### Data Preparation

Download MoleculeNet datasets [here](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip), unzip the file, and put the dataset fold under `datasets/dp/`. You can use the following commands from within `OpenBioMed/`:

```shell
wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
unzip chem_dataset.zip
mkdir -p datasets/dp
mv dataset datasets/dp/moleculenet
rm chem_dataset.zip
```

After downloading and unzipping, you should remove all the `processed/` directories of 8 datasets in the `dataset/` folder. Otherwise you will get the following error:
```shell
RuntimeError: The 'data' object was created by an older version of PyG. If this error occurred while loading an already existing dataset, remove the 'processed/' directory in the dataset's root folder and try again.
```

#### Model Preparation

To reproduce DeepEIK, you should install PubMedBERT (uncased) from [huggingface](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext) and put the checkpoint under `ckpts/text_ckpts/`. 

To reproduce or finetune MolCLR, MoMu and GraphMVP used their trained checkpoints, you can download the checkpoints from:  
MolCLR: https://github.com/yuyangw/MolCLR  
MoMu: https://github.com/ddz16/MoleculePrediction  
GraphMVP: https://github.com/chao1224/GraphMVP  

You will need to rename and put the checkpoints at the following paths.
```shell
# MolCLR
ckpts/gnn_ckpts/molclr/model.pth
# Momu
ckpts/fusion_ckpts/momu/MoMu-K.ckpt
# GraphMVP
ckpts/gnn_ckpts/graphmvp/pretraining_model.pth
```

#### Training and Evaluation

You can run scripts using bash under `scripts/dp/`:

```bash
scripts/dp
├── train_molclr.sh		# running MolCLR on 8 datasets of  moleculenet
├── train_graphmvp.sh  # running GraphMVP on 8 datasets of moleculenet
├── train_momu.sh      # running MoMu on 8 datasets of  moleculenet
└── train_deepeik.sh   # running DeepEIK on 8 datasets of moleculenet
```

You can also modify the scripts or directly use the following command:

```bash
python tasks/mol_task/dp.py \
[--device DEVICE] \									  # gpu device id
[--mode MODE] \											  # training mode, train: train-test
[--config_path CONFIG_PATH] \				  # configuration file, see configs/dti/ for more details
[--dataset DATASET] \								  # datasets name, support MoleculeNet now
[--dataset_path DATASET_PATH] \       # path to the datasets
[--dataset_name DATASET_NAME] \       # name of the dataset
[--init_checkpoint INIT_CHECKPOINT] \ # checkpoint path used for efficient validation
[--param_key PARAM_KEY] \							# key of the checkpoint dict that contains model parameters
[--output_path OUTPUT_PATH] \         # save checkpoint path used for training
[--num_workers NUM_WORKERS] \         # number of workers when loading data
[--patience PATIENCE] \               # number of tolerant epochs for early-stopping
[--weight_decay WEIGHT_DECAY] \       # weight decay, default is 1e-5
[--lr LR] \                           # learning rate, default is 1e-3
[--batch_size BATCH_SIZE] \           # batch size, default is 128
[--epochs EPOCHS] \                   # number of training epochs
[--logging_steps LOGGING_STEPS] \     # steps for printing training information
[--seed SEED]                         # random seed
[--dropout DROPOUT]                   # The dropout ratio of dp model
```

