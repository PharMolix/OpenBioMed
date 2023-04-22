## Drug-Target Interaction Prediction

Drug-target interaction prediction aims to predict the binding relationships between drugs and target proteins, which is a significant task in AI drug discovery.

#### Features

- Supported models: DeepDTA, MGraphDTA and DeepEIK. More models will be implemented and more combinations will be tested in the future.
- Supported datasets: 2 classification datasets i.e. Yamanishi08's and BMKG-DTI, 2 regression datasets i.e. Davis and KIBA.
- Supported split: warm start; cold-drug; cold-protein; cold-cluster;
- Supproted evaluation: ROC_AUC, PR_AUC, F1, Precision and Recall for classification; MSE, Pearson and Spearman coefficient, CI index and $r_m^2$ index for regression.

#### Additional Packages

To get full access to DTI functions (all splitting strategies and evaluation metrics), run the following:

```bash
# NOTE: Make sure you are at OpenBioMed/ directory
pip install lifelines

# NOTE: ONLY RUN THE FOLLOWING SECTION IF YOU HAVE NOT YET INSTALLED
# PyBioMed
#-------------------------------------------------------------------
cd assets # create this directory if not yet existent
# PyBioMed for calculating protein descriptors
git clone https://github.com/gadsbyfly/PyBioMed
cd PyBioMed
python setup.py install
cd ..
#-------------------------------------------------------------------

# It is recommended to install cogdl for calculating network embeddings if you want to reproduce DeepEIK
git clone https://github.com/THUDM/cogdl
cd cogdl
pip install -e .
```

#### Data Preparation

Davis and KIBA can download from [DeepDTA](https://github.com/hkmztrk/DeepDTA/tree/master/data), Yamanishi08 and BMKG_DTI can download from [here](​https://drive.google.com/drive/folders/1AaUWLlOOua5BH7Q-bBVUBgOugDfWF3ip?usp=sharing). The 4 datasets should put under `datasets/dti/`. It is recommended to install BMKGv1 [here](​https://drive.google.com/drive/folders/1U2M3383-3dDAyLTAcXGcUagAEjlB6QgN?usp=sharing
) and put it under `assets/kg/`.

#### Model Preparation

To reproduce DeepEIK, you should install PubMedBERT (uncased) from [huggingface](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext) and put the checkpoint under `ckpts/text_ckpts/`. 

#### Training and Evaluation

You can run scripts using bash under `scripts/dti/`:

```bash
scripts/dti
├── train_baseline.sh             # running DeepDTA or MGraphDTA on Yamanishi08's and BMKG-DTI under 4 settings
├── train_baseline_regression.sh  # running DeepDTA or MGraphDTA on Davis or KIBA
├── train_deepeik.sh              # running DeepEIK on Yamanishi08's and BMKG-DTI under 4 settings
└── train_deepeik_regression.sh   # running DeepEIK on Davis or KIBA
```

You can also modify the scripts or directly use the following command:

```bash
python tasks/mol_task/dti.py \
--device DEVICE \                   # gpu device id
--mode MODE \                       # training mode, kfold: 5-fold validation, train: train-test
--config_path CONFIG_PATH \         # configuration file, see configs/dti/ for more details
--dataset DATASET \                 # dataset name, select from [yamanishi08, bmkg-dti, davis, kiba]
--dataset_path DATASET_PATH \       # path to the dataset
--split_strategy SPLIT_STRATEGY \   # split strategy, select from [warm, cold-drug, cold-protein, cold-cluster]
--init_checkpoint INIT_CHECKPOINT \ # checkpoint path used for efficient validation
--param_key PARAM_KEY \             # key of the checkpoint dict that contains model parameters
--output_path OUTPUT_PATH \         # save checkpoint path used for training
--num_workers NUM_WORKERS \         # number of workers when loading data
--epochs EPOCHS \                   # number of training epochs
--patience PATIENCE \               # number of tolerant epochs for early-stopping
--weight_decay WEIGHT_DECAY \       # weight decay, default is 1e-5
--lr LR \                           # learning rate, default is 1e-3
--batch_size BATCH_SIZE \           # batch size, default is 128
--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS \ # steps for gradient accumulation
--logging_steps LOGGING_STEPS \     # steps for printing training information
--seed SEED                         # random seed
```

