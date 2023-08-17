# OpenBioMed
This repository holds OpenBioMed, an open-source toolkit for multi-modal representation learning in AI-driven biomedical research. Our focus is on multi-modal information, e.g. knowledge graphs and biomedical texts for drugs, proteins, and single cells, as well as a wide range of applications, including drug-target interaction prediction, molecular property prediction, cell-type prediction, molecule-text retrieval, molecule-text generation, and drug-response prediction. **Researchers can compose a large number of deep learning models including LLMs like BioMedGPT-1.6B and CellLM to facilitate downstream tasks.** We provide easy-to-use APIs and commands to accelerate life science research.

## News!

- [04/23] 🔥The pre-alpha BioMedGPT model and OpenBioMed are available!
- [06/12] 🔥The paper of CellLM is now avaliable on arxiv, and the latest checkpoint of CellLM has been updated on the cloud drive!

## Features

- **3 different modalities for drugs, proteins, and cell-lines**: molecular structure, knowledge graphs, and biomedical texts. We present a unified and easy-to-use pipeline to load, process, and fuse the multi-modal information.
- **BioMedGPT-1.6B, including other 20 deep learning models**, ranging from CNNs and GNNs to Transformers. **BioMedGPT-1.6B** is a pre-trained multi-modal molecular foundation model with 1.6B parameters that associates 2D molecular graphs with texts. We also present **CellLM**, a single cell foundation model with 50M parameters.
- The checkpoints of BioMedGPT-1.6B and CellLM can be downloaded from [here](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg) (password is `7a6b`). You can test the performance of BioMedGPT-1.6B on molecule-text retrieval by running `scripts/mtr/run.sh`, or test the performance of CellLM on cell type classification by running `scripts/ctc/train.sh`.
- **8 downstream tasks** including AIDD tasks like drug-target interaction prediction and molecule property training, as wel as cross-modal tasks like molecule captioning and text-based molecule generation.  
- **20+ datasets** that are most popular in AI-driven biomedical research. Reproductible benchmarks with abundant model combinations and comprehensive evaluations are provided.
- **3 knowledge graphs** with extensive domain expertise. We present **BMKGv1**, a knowledge graph containing 6,917 drugs, 19,992 proteins, and 2,223,850 relationships with text descriptions. We offer APIs to load and process these graphs and link drugs and proteins based on structural information.

## Installation

To support basic usage of OpenBioMed, run the following commands:

```bash
conda create -n OpenBioMed python=3.8
conda activate OpenBioMed
conda install -c conda-forge rdkit

pip install torch
conda install pytorch-cluster -c pyg
conda install pytorch-scatter -c pyg
conda install pytorch-sparse -c pyg
conda install pytorch-spline-conv -c pyg
pip install torch-geometric
# If you have issues installing the above PyTorch-related packages, instructions at https://pytorch.org/get-started/locally/
# and https://github.com/pyg-team/pytorch_geometric may help. You may find it convenient to directly install PyTorch
# Geometric and its extensions from wheels available at https://data.pyg.org/whl/.

pip install transformers
pip install ogb
git clone https://github.com/BioFM/OpenBioMed.git # this repository
cd OpenBioMed
mkdir assets
mkdir ckpts
```

**Note** that additional packages may be required for specific downstream tasks.

## Quick Start

Here, we provide a quick example of training DeepDTA for drug-target interaction prediction on the Davis dataset. For more models, datasets, and tasks, please refer to our [scripts](./open_biomed/scripts) and [documents](./docs).

This quick example requires installation of an additional package:
```bash
cd assets
git clone https://github.com/gadsbyfly/PyBioMed.git
cd PyBioMed
python setup.py install
cd ..
```

### Step 1: Data Preparation

Download the Davis dataset [here](https://github.com/hkmztrk/DeepDTA/tree/master/data) and run the following from `OpenBioMed/` directory (the top level of this repository):

```
mkdir datasets
cd datasets
mkdir dti
mv [your_path_of_davis] ./dti/davis
```

### Step 2: Training and Evaluation

Run:

```bash
cd ../open_biomed
bash scripts/dti/train_baseline_regression.sh
```

The results will look like the following (running takes around 40 minutes on an NVIDIA A100 GPU):

```bash
INFO - __main__ - MSE: 0.2198, Pearson: 0.8529, Spearman: 0.7031, CI: 0.8927, r_m^2: 0.6928
```

## Contact Us

As a pre-alpha version release, we are looking forward to user feedback to help us improve our framework. If you have any questions or suggestions, please open an issue or contact [dair@air.tsinghua.edu.cn](mailto:dair@air.tsinghua.edu.cn).


## Cite Us

If you find our open-sourced code & models helpful to your research, please consider giving this repo a star🌟 and citing📑 the following article. Thank you for your support!
```
@misc{OpenBioMed_code,
  author={Luo, Yizhen and Yang, Kai and Hong, Massimo and Liu, Xing Yi and Zhao, Suyuan and Zhang, Jiahuan and Wu, Yushuai and Nie, Zaiqing},
  title={Code of OpenBioMed},
  year={2023},
  howpublished={\url{https://github.com/BioFM/OpenBioMed.git}}
}
```
```
@misc{luo2023empowering,
      title={Empowering AI drug discovery with explicit and implicit knowledge}, 
      author={Yizhen Luo and Kui Huang and Massimo Hong and Kai Yang and Jiahuan Zhang and Yushuai Wu and Zaiqing Nie},
      year={2023},
      eprint={2305.01523},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
```
@misc{zhao2023largescale,
      title={Large-Scale Cell Representation Learning via Divide-and-Conquer Contrastive Learning}, 
      author={Suyuan Zhao and Jiahuan Zhang and Zaiqing Nie},
      year={2023},
      eprint={2306.04371},
      archivePrefix={arXiv},
      primaryClass={cs.CE}
}
```

## Contributing

If you encounter problems using OpenBioMed, feel free to create an issue! 
