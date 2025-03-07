<div align="center"><h1>OpenBioMed</h1></div>
<h4 align="center">
    <p>
        <b>English</b> |
        <a href="./README-CN.md">中文</a>
    <p>
</h4>

## News 🎉


- [2025/03/07] 🔥 Released **OpenBioMed v2**. We present new features including additional downstream biomedical tasks, more flexible data APIs, and a user-friendly [online platform](http://openbiomed.pharmolix.com) for customizing workflows and LLM agents in solving complicated scientific research tasks. We also release a preview version of **PharmolixFM** (📃[Paper](), 🤖[Model](https://cloud.tsinghua.edu.cn/f/8f337ed5b58f45138659/), 📎[Citation](#to-cite-pharmolixfm)). BioMedGPT-R1 inference is currently supported, and fine-tuning will be available in this month!

> PharmolixFM is an all-atom molecular foundation model jointly released by PharMolix Inc. and Institute of AI Industry Research (AIR), Tsinghua University. It unifies molecules, antibodies, and proteins by jointly modeling them at atom-level with cutting-edge non-autoregressive multi-modal generative models. PharmolixFM is capable of solving mutiple downstream tasks such as docking, structure-based drug design, peptide design, and molecular conformation generation. PharmolixFM achieves competitive performance with AlphaFold3 (83.9 vs 90.2, RMSD < 2Å) on protein-molecule docking (given pocket).


- [2025/02/20] 🔥 BioMedGPT-R1 (🤗[Huggingface Model](https://huggingface.co/PharMolix/BioMedGPT-R1)) has been released. 

> BioMedGPT-R1-17B is a multimodal biomedical reasoning model jointly released by PharMolix and Institute of AI Industry Research (AIR) . It updates the language model in last version with DeepSeek-R1-Distill-Qwen-14B and adopts two-stage training for cross-modal alignment and multimodal reasoning SFT, performing on par with commercial model on biomedical QA benchmark.

- [2024/05/16] Released implementation of **LangCell** (📃[Paper](https://arxiv.org/abs/2405.06708), 💻[Code](https://github.com/PharMolix/LangCell), 🤖[Model](https://drive.google.com/drive/folders/1cuhVG9v0YoAnjW-t_WMpQQguajumCBTp?usp=sharing), 📎[Citation](#to-cite-langcell)).

> LangCell is the first "language-cell" multimodal pre-trained model jointly developed by PharMolix and Institute for AI Industry Research (AIR). It effectively enhances the understanding of single-cell transcriptomics by learning knowledge-rich texts containing cell identity information, and addresses the task of cell identity understanding in data-scarce scenarios. LangCell is the only single-cell model capable of effective zero-shot cell identity understanding and has also achieved SOTA in few-shot and fine-tuning scenarios. LangCell will soon be integrated into OpenBioMed. 


- [2023/08/14] Released implementation of **BioMedGPT-10B** (📃[Paper](https://arxiv.org/abs/2308.09442v2), 🤖[Model](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F), 📎[Citation](#to-cite-biomedgpt)), **BioMedGPT-LM-7B** (🤗[HuggingFace Model](https://huggingface.co/PharMolix/BioMedGPT-LM-7B)) and **DrugFM** (🤖[Model](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F)).

> BioMedGPT-10B is the first commercial-friendly multimodal biomedical foundation model jointly released by PharMolix and Institute of AI Industry Research (AIR). It aligns the language of life (molecular structures and protein sequences) with human natural language, performing on par with human experts on biomedical QA benchmarks, and demonstrating powerful performance in cross-modal molecule and protein question answering tasks. BioMedGPT-LM-7B is the first commercial-friendly generative foundation model tailored for biomedicine based on Llama-2. 

> DrugFM is a multi-modal molecular foundation model jointly developed by Institute of AI Industry Research (AIR) and Beijing Academy of Artificial Intelligence, BAAI. It leverages UniMAP, a pre-trained molecular model that captures fine-grained properties and representations of molecules, and incorporates MolFM, our multimodal molecular foundation model. DrugFM achieves SOTA on cross-modal retrieval.


- [2023/06/12] Released implementation of **MolFM** (📃[Paper](https://arxiv.org/abs/2307.09484), 🤖[Model](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F), 📎[Citation](#to-cite-molfm)) and **CellLM** (📃[Paper](https://arxiv.org/abs/2306.04371), 🤖[Model](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg), 📎[Citation](#to-cite-celllm)). 

> MolFM is a multi-modal molecular foundation model that enables joint comprehension of molecular structures, biomedical documents and knowledge graphs. On cross-modal retrieval, MolFM outperforms existing models by 12.03% and 5.04% under zero-shot and fine-tuning settings. MolFM also excels in molecule captioning, text-to-molecule generation and molecule property prediction.

> CellLM is the first large-scale cell representation learning model trained on both normal cells and cancer cells with divide-and-conquer contrastive learning. CellLM beats ScBERT on cell type annotation (71.8 vs 68.8), few-shot single-cell drug sensitivity prediction (88.9 vs 80.6) and single-omics cell line drug sensitivity prediction (93.4 vs 87.2).


- [2023/04/23] Released implementation of **BioMedGPT-1.6B** (🤖[Model](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg)) and **OpenBioMed**.


## Table of contents


- [Introduction](#introduction)
- [Installation](#installation)
- [Tutorials](#tutorials)
- [Previous version](#previous-version)
- [Limitations](#limitations)
- [Cite us](#cite-us)


## Introduction


This repository holds OpenBioMed, a Python deep learning toolkit for AI-empowered biomedicine. OpenBioMed provides **flexible APIs to handle multi-modal biomedical data**, including molecules, proteins, single cells, natural language, and knowledge graphs. OpenBioMed builds **20+ tools that covers a wide range of downstream applications**, ranging from traditional AI drug discovery tasks to newly-emerged multi-modal challenges. Moreover, OpenBioMed provides **an easy-to-use interface for building workflows** that connect multiple tools and developing LLM-driven agents for solving complicated biomedical research tasks.


OpenBioMed provide researchers with access to:


- **4 types of data modalities**:  OpenBioMed provide easy-to-use APIs for researchers to access and process different types of data including molecules, proteins, pockets, and texts. New data structures for DNAs, RNAs, single cells, and knowledge graphs will be available in future versions.
- **20+ tools**, ranging from ML-based prediction models for AIDD tasks including molecule property prediction and protein folding, to visualization tools and web-search APIs. 
- **20+ deep learning models**, comprising exclusive models such as [PharmolixFM](https://cloud.tsinghua.edu.cn/f/8f337ed5b58f45138659/), [BioMedGPT-R1](https://huggingface.co/PharMolix/BioMedGPT-R1), [BioMedGPT](https://ieeexplore.ieee.org/document/10767279/) and [MutaPLM](https://arxiv.org/abs/2410.22949).


Key features of OpenBioMed include:


- **Unified Data Processing Pipeline**: easily load and transform the heterogeneous data from different biomedical entities and modalities into a unified format.
- **Off-the-shelf Inference**: publicly available pre-trained models and inference demos, readily to be transferred to your own data or task. 
- **Easy-to-use Interface for Building Workflows and LLM Agents**: flexibly build solutions for complicated research tasks with multi-tool collaborative workflows, and harvest LLMs for simulating trial-and-errors and gaining scientific insights.


Here is a list of currently available tools. This is a continuing effort and we are working on further growing the list.


|              Tool              |                       Supported Model                        |                         Description                          |
| :----------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Molecular Property Prediction  |         [GraphMVP](https://arxiv.org/abs/2110.07728)         | Predicting the properties of a given molecule (e.g. blood-brain barrier penetration and side effects) |
|  Molecule Question Answering   |          [BioT5](https://arxiv.org/abs/2310.07276)           | Answering textual queries of a given molecule (e.g. structural descriptions, functional groups, number of hydrogen bond donors) |
|     Molecule Visualization     |                             N/A                              |                     Visualize a molecule                     |
|    Molecule Name/ID Request    |                             N/A                              | Obtaining a molecule from PubChem using its name or PubChemID |
|   Molecule Structure Request   |                             N/A                              | Obtaining a molecule from PubChem based on similar structures |
|   Protein Question Answering   |          [BioT5](https://arxiv.org/abs/2310.07276)           | Answering textual queries of a given protein (e.g. motifs, functions, subcellular location, related diseases) |
|        Protein Folding         | [ESMFold](https://www.science.org/doi/10.1126/science.ade2574) | Predicting the 3D structure of a protein based on its amino acid sequence |
|   Protein Pocket Prediction    | [P2Rank](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0285-8) |     Predicting potential binding sites within a protein      |
|      Mutation Explanation      |         [MutaPLM](https://arxiv.org/abs/2410.22949)          | Providing textual explanations of a single-site substitution mutation on a protein sequence |
|      Mutation Engineering      |         [MutaPLM](https://arxiv.org/abs/2410.22949)          | Generating a mutated protein to fit the textual instructions on the wild-type protein sequence. |
|   Protein UniProtID Request    |                             N/A                              | Obtaining a protein sequence from UniProtKB based on UniProt accession ID |
|      Protein PDB Request       |                             N/A                              | Obtaining a protein structure from PDB/AlphaFoldDB based on PDB/AlphaFoldDB accession ID |
|     Protein Visualization      |                             N/A                              |                     Visualize a protein                      |
| Protein-molecule Rigid Docking | [PharmolixFM](https://cloud.tsinghua.edu.cn/f/8f337ed5b58f45138659/) | Generate the binding pose of the molecule with a given pocket in a protein |
|  Structure-based Drug Design   | [PharmolixFM](https://cloud.tsinghua.edu.cn/f/8f337ed5b58f45138659/) | Generate a molecule that binds with a given pocket in a protein |
|     Complex Visualization      |                             N/A                              |             Visualize a protein-molecule complex             |
|      Pocket Visualization      |                             N/A                              |             Visualize a pocket within a protein              |
|          Web Request           |                             N/A                              |             Obtaining information by web search              |

## Installation

To enable basic features of OpenBioMed, please execute the following:


```bash
conda create -n OpenBioMed python=3.9
conda activate OpenBioMed
pip install torch==1.13.1+{your_cuda_version} torchvision==0.14.1+{your_cuda_version} torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/{your_cuda_version}  
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+{your_cuda_version}.html
pip install pytorch_lightning==2.0.8 peft==0.9.0 accelerate==1.3.0 --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt
```


We recommend using cuda=11.7 to set up the environment. Other versions of cudatoolkits may lead to unexpected problems.


To enable visualization tools and vina score computation tools, you should install the following packages:

```
# For visualization
conda install -c conda-forge pymol-open-source
pip install imageio

# For AutoDockVina
pip install meeko==0.1.dev3 pdb2pqr vina==1.2.2 
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3

# For overlap-based evaluation
pip install spacy rouge_score nltk
python
>>> import nltk
>>> nltk.download('wordnet')
>>> nltk.download('omw-1.4')
```


## Tutorials

Checkout our [Jupytor notebooks](./examples/) for a quick start!

| Name                                                         | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [BioMedGPT Inference](./examples/biomedgpt_r1.ipynb)         | Examples of using BioMedGPT-10B to answer questions about molecules and proteins and BioMedGPT-R1 to perform reasoning. |
| [Molecule Processing](./examples/manipulate_molecules.ipynb) | Examples of using OpenBioMed APIs to load, process, and export molecules and proteins. |
| [ML Tool Usage](./examples/explore_ai4s_tools.ipynb)         | Examples of using machine learning tools to perform inference. |
| [Visualization](./examples/visualization.ipynb)              | Examples of using OpenBioMed APIs to visualize molecules, proteins, complexes, and pockets. |
| [Workflow Construction](./examples/workflow.ipynb)           | Examples of building and executing workflows and developing LLM agents for complicated scientific tasks. |
| [Model Customization](./examples/model_customization.ipynb)  | Tutorials on how to customize your own model and data using OpenBioMed training pipelines. |

## Previous Version

If you hope to use the features of the previous version, please switch to the `v1.0` branch of this repository by running the following command:

```bash
git checkout v1.0
```

## Limitations

This repository holds BioMedGPT-LM-7B, BioMedGPT-10B, and BioMedGPT-R1, and we emphasize the responsible and ethical use of these models. BioMedGPT should NOT be used to provide services to the general public. Generating any content that violates applicable laws and regulations, such as inciting subversion of state power, endangering national security and interests, propagating terrorism, extremism, ethnic hatred and discrimination, violence, pornography, or false and harmful information, etc. is strictly prohibited. BioMedGPT is not liable for any consequences arising from any content, data, or information provided or published by users.

## License

This repository is licensed under the [MIT License](./LICENSE). The use of BioMedGPT-LM-7B and BioMedGPT-10B models is accompanied with [Acceptable Use Policy](./USE_POLICY.md).

## Contact Us

We are looking forward to user feedback to help us improve our framework. If you have any technical questions or suggestions, please feel free to open an issue. For commercial support or collaboration, please contact [opensource@pharmolix.com](mailto:opensource@pharmolix.com).


## Cite Us

If you find our open-sourced code and models helpful to your research, please consider giving this repository a 🌟star and 📎citing our research papers. Thank you for your support!

##### To cite OpenBioMed:

```
@misc{OpenBioMed_code,
      author={Luo, Yizhen and Yang, Kai and Fan, Siqi and Hong, Massimo and Nie, Zikun and Luo, Wen and Xie, Ailin and Liu, Xing Yi and Zhao, Suyuan and Zhang, Jiahuan and Wu, Yushuai and Nie, Zaiqing},
      title={Code of OpenBioMed},
      year={2023},
      howpublished={\url{https://github.com/Pharmolix/OpenBioMed.git}}
}
```

##### To cite BioMedGPT:

```
@article{luo2024biomedgpt,
  title={Biomedgpt: An open multimodal large language model for biomedicine},
  author={Luo, Yizhen and Zhang, Jiahuan and Fan, Siqi and Yang, Kai and Hong, Massimo and Wu, Yushuai and Qiao, Mu and Nie, Zaiqing},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2024},
  publisher={IEEE}
}
```

##### To cite PharmolixFM:

Coming soon!

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

##### To cite LangCell:

```
@misc{zhao2024langcell,
      title={LangCell: Language-Cell Pre-training for Cell Identity Understanding}, 
      author={Suyuan Zhao and Jiahuan Zhang and Yizhen Luo and Yushuai Wu and Zaiqing Nie},
      year={2024},
      eprint={2405.06708},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN}
}
```

##### To cite MutaPLM:

```
@article{luo2025mutaplm,
  title={MutaPLM: Protein Language Modeling for Mutation Explanation and Engineering},
  author={Luo, Yizhen and Nie, Zikun and Hong, Massimo and Zhao, Suyuan and Zhou, Hao and Nie, Zaiqing},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={79783--79818},
  year={2025}
}
```