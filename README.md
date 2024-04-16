<div align="center"><h1>DrugFM</h1></div>

This repository mainly holds **DrugFM** (🤖[Model](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F)), a multi-modal molecular foundation model jointly developed by Institute of AI Industry Research (AIR) and Beijing Academy of Artificial Intelligence, BAAI. DrugFM comprises **1.06B** parameters. It leverages a MoE gate to jointly incorporate molecular representations from GraphMVP, UniMAP and UniMol based on text features. It leverages a multi-modal encoder and a multi-modal decoder for jointly comprehending molecules and texts. DrugFM achieves SOTA on cross-modal retrieval and generation.

## Installation

1. (Optional) Creating conda environment:

```bash
conda create -n drugfm python=3.9
conda activate drugfm
```

2. Install required packages:

```
pip install -r requirements.txt
```

3. Install Pyg dependencies:

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-(your_torch_version)+(your_cuda_version).html
pip install torch-geometric
# If you have issues installing the above PyTorch-related packages, instructions at https://pytorch.org/get-started/locally/ and https://github.com/pyg-team/pytorch_geometric may help. You may find it convenient to directly install PyTorch Geometric and its extensions from wheels available at https://data.pyg.org/whl/.
```

4. Install packages for cross-modal generation

```bash
pip install spacy
pip install rouge_score
pip install Levenshtein

pip install nltk
python
>>> import nltk
>>> nltk.download('wordnet')
>>> nltk.download('omw-1.4')
```


## Quick Start

For cross-modal retrieval, run the following script:
```bash
bash scripts/multimodal/mtr/test_pcdes.sh drugfm cuda:0 # switch to your own cuda device or cpu
```

For molecule captioning, run the following script:
```bash
bash scripts/multimodal/molcap/train.sh drugfm cuda:0 # switch to your own cuda device or cpu
```

For molecule captioning, run the following script:
```bash
bash scripts/multimodal/text2smi/train.sh drugfm cuda:0 # switch to your own cuda device or cpu
```

## Cite DrugFM

```
@misc{DrugFM_code,
      author={Luo, Yizhen and Yang, Kai and Hong, Massimo and Liu, Xing Yi and Zhao, Suyuan and Zhang, Jiahuan and Wu, Yushuai and Nie, Zaiqing},
      title={Code of DrugFM},
      year={2023},
      howpublished={\url{https://github.com/Pharmolix/OpenBioMed.git}}
}
```
