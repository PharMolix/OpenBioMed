
<div align="center"><h1>OpenBioMed</h1></div>
<h4 align="center">
    <p>
        <b>English</b> |
        <a href="./README-CN.md">中文</a>
    <p>
</h4>

## News 🎉

- [08/14] 🔥 Released implementation of **BioMedGPT-10B** (📃[Technical Report](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F), 🤖[Model](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F), 📎[Cite](#to-cite-biomedgpt)), **BioMedGPT-LM-7B** (🤗[HuggingFace Model](https://huggingface.co/PharMolix/BioMedGPT-LM-7B)) and **DrugFM** (🤖[Model](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F)) .

    > BioMedGPT-10B is the first commercial-friendly multimodal biomedical foundation model jointly released by PharMolix and Institute of AI Industry Research (AIR) . It aligns the language of life (molecular structures and protein sequences) with human natural language, performing on par with human experts on biomedical QA benchmarks, and demonstrating powerful performance in cross-modal molecule and protein question answering tasks. BioMedGPT-LM-7B is the first commercial-friendly generative foundation model tailored for biomedicine based on Llama-2. 

    > DrugFM is a multi-modal molecular foundation model jointly developed by Institute of AI Industry Research (AIR) and Beijing Academy of Artificial Intelligence, BAAI. It leverages UniMAP, a pre-trained molecular model that captures fine-grained properties and representations of molecules, and incorporates MolFM, our multimodal molecular foundation model. DrugFM achieves SOTA on cross-modal retrieval.

- [06/12] Released implementation of **MolFM** (📃[Paper](https://arxiv.org/abs/2307.09484), 🤖[Model](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F), 📎[Cite](#to-cite-molfm)) and **CellLM** (📃[Paper](https://arxiv.org/abs/2306.04371), 🤖[Model](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg), 📎[Cite](#to-cite-celllm)). 

    > MolFM is a multi-modal molecular foundation model that enables joint comprehension of molecular structures, biomedical documents and knowledge graphs. On cross-modal retrieval, MolFM outperforms existing models by 12.03% and 5.04% under zero-shot and fine-tuning settings. MolFM also excels in molecule captioning, text-to-molecule generation and molecule property prediction.

    > CellLM is the first large-scale cell representation learning model trained on both normal cells and cancer cells with divide-and-conquer contrastive learning. CellLM beats ScBERT on cell type annotation (71.8 vs 68.8), few-shot single-cell drug sensitivity prediction (88.9 vs 80.6) and single-omics cell line drug sensitivity prediction (93.4 vs 87.2).

- [04/23] Released implementation of **BioMedGPT-1.6B** (🤖[Model](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg)) and **OpenBioMed**.

## Table of contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentations](./docs)
- [Limitations](#limitations)
- [Cite us](#cite-us)

## Introduction

This repository holds OpenBioMed, a Python deep learning toolkit for AI-empowered biomedicine. OpenBioMed provides easy access to multimodal biomedical data, i.e. molecular structures, transcriptomics, knowledge graphs and biomedical texts for molecules, proteins, and single cells. OpenBioMed supports a wide range of downstream applications, ranging from traditional AI drug discovery tasks to newly-emerged multimodal challenges. 

OpenBioMed provide researchers with easy-to-use APIs to:

- **3 different modalities for molecules, proteins, and single cells**: molecular structures or transcriptomics, biomedical texts, and knowledge graphs. OpenBioMed provide a unified pipeline for researchers to access, process, and fuse these modalities.
- **10 downstream tasks**, categorized into AI drug discovery (AIDD) tasks like drug-target binding affinity prediction and molecule property prediction, as well as multimodal tasks like molecule captioning and text-based molecule generation. 
- **20+ deep learning models**, including [BioMedGPT-10B](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F), [MolFM](https://arxiv.org/abs/2307.09484), [CellLM](https://arxiv.org/abs/2306.04371). Researchers can flexibly compose different components to curate their own models.
- **20+ datasets** that are most popular in AI-driven biomedical research. Reproductible benchmarks with abundant model combinations and comprehensive evaluations are provided.

Key features of OpenBioMed include:

- **Unified Data Processing Pipeline**: easily load and transform the heterogeneous data from different biomedical entities and modalities into a unified format.
- **Off-the-shelf Inference**: publicly available pre-trained models and inference demos, readily to be transferred to your own data or task. 
- **Reproducible Model Zoo**: flexibly replicate and extend state-of-the-art models on existing and new applications.

The following table shows the supported tasks, datasets and models in OpenBioMed. This is a continuing effort and we are working on further growing the list.

|                  Task                   |         Supported Datasets         |              Supported Models               |
| :-------------------------------------: | :--------------------------------: | :-----------------------------------------: |
|          Cross-modal Retrieval          |               PCdes                |   KV-PLM, SciBERT, MoMu, GraphMVP, MolFM    |
|           Molecule Captioning           |              ChEBI-20              |   MolT5, MoMu, GraphMVP, MolFM, BioMedGPT   |
|     Text-based Molecule Generation      |              ChEBI-20              |         MolT5, SciBERT, MoMu, MolFM         |
|       Molecule Question Answering       |             ChEMBL-QA              |           MolT5, MolFM, BioMedGPT           |
|       Protein Question Answering        |             UniProtQA              |                  BioMedGPT                  |
|        Cell Type Classification         |          Zheng68k, Baron           |               scBERT, CellLM                |
|  Single Cell Drug Response Prediction   |                GDSC                |            DeepCDR, TGSA, CellLM            |
|      Molecule Property Prediction       |            MoleculeNet             | MolCLR, GraphMVP, MolFM, DeepEIK, BioMedGPT |
| Drug-target Binding Affinity Prediction | Yamanishi08, BMKG-DTI, DAVIS, KIBA |         DeepDTA, MGraphDTA, DeepEIK         |
| Protein-protein Interaction Prediction  |      SHS27k, SHS148k, STRING       |         PIPR, GNN-PPI, OntoProtein          |


## Installation

1. (Optional) Creating conda environment:

```bash
conda create -n OpenBioMed python=3.9
conda activate OpenBioMed
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

**Note**: additional packages may be required for some downstream tasks.

## Quick Start

Checkout our [Jupytor notebooks](./examples) and [documentations](./docs) for a quick start!

| Name                                                         | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [BioMedGPT-10B Inference](./examples/biomedgpt_inference.ipynb) | Example of using BioMedGPT-10B to answer questions about molecules and proteins. |
| [Cross-modal Retrieval with MolFM](./examples/cross_modal_retrieval) | Example of using MolFM to retrieve the most related text descriptions for a molecule. |
| [Text-based Molecule Generation with MolT5](./examples/molecule_generation.ipynb) | Example of using MolT5 to generate the SMILES string of a molecule based on text description. |
| [Cell Type classification with CellLM](./examples/cell_type_classification.ipynb) | Example of using fine-tuned CellLM to classify cell types.   |
| [Molecule Property prediction](./docs/dp.md)                 | Training & testing pipeline of the molecule propery prediction task |
| [Drug-response prediction](./docs/drp.md)                    | Training & testing pipeline of the drug-response prediction task |
| [Drug-target binding affinity prediction](./docs/dti.md)     | Training & testing pipeline of the drug-target binding affinity prediction task |
| [Molecule captioning](./docs/molcap.md)                      | Training & testing pipeline of the molecule captioning task  |

## Limitations

This repository holds BioMedGPT-LM-7B and BioMedGPT-10B, and we emphasize the responsible and ethical use of these model. BioMedGPT should **NOT** be used to provide services to the general public. Generating any content that violates applicable laws and regulations, such as inciting subversion of state power, endangering national security and interests, propagating terrorism, extremism, ethnic hatred and discrimination, violence, pornography, or false and harmful information, etc. is strictly prohibited. BioMedGPT is not liable for any consequences arising from any content, data, or information provided or published by users.

## License

This repository is licensed under the [MIT License](./LICENSE). The use of BioMedGPT-LM-7B and BioMedGPT-10B models is accompanied with [Acceptable Use Policy](./USE_POLICY.md).

## Contact Us

We are looking forward to user feedback to help us improve our framework. If you have any technical questions or suggestions, please feel free to open an issue. For commercial support or collaboration, please contact [opensource@pharmolix.com](mailto:opensource@pharmolix.com).


## Cite Us

If you find our open-sourced code and models helpful to your research, please consider giving this repository a 🌟star and 📎citing the following articles. Thank you for your support!

##### To cite OpenBioMed:

```
@misc{OpenBioMed_code,
      author={Luo, Yizhen and Yang, Kai and Hong, Massimo and Liu, Xing Yi and Zhao, Suyuan and Zhang, Jiahuan and Wu, Yushuai and Nie, Zaiqing},
      title={Code of OpenBioMed},
      year={2023},
      howpublished={\url{https://github.com/BioFM/OpenBioMed.git}}
}
```

##### To cite BioMedGPT:

Please wait for our upcoming updates on Arxiv.

##### To cite DeepEIK:

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
##### To cite MolFM:
```
@misc{luo2023molfm,
      title={MolFM: A Multimodal Molecular Foundation Model}, 
      author={Yizhen Luo and Kai Yang and Massimo Hong and Xing Yi Liu and Zaiqing Nie},
      year={2023},
      eprint={2307.09484},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM}
}
```

##### To cite CellLM:
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
