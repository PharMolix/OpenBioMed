<div align="center"><h1>OpenBioMed</h1></div>
<h4 align="center">
    <p>
        <b>English</b> |
        <a href="./README-CN.md">中文</a>
    <p>
</h4>

[![GitHub Repo stars](https://img.shields.io/github/stars/PharMolix/OpenBioMed?style=social)](https://github.com/PharMolix/OpenBioMed/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/PharMolix/OpenBioMed)](https://github.com/PharMolix/OpenBioMed/commits/main)
[![GitHub contributors](https://img.shields.io/github/contributors/PharMolix/OpenBioMed?color=orange)](https://github.com/PharMolix/OpenBioMed/graphs/contributors)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/PharMolix/OpenBioMed/pulls)
[![Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/PharMolix)
[![Docker Pulls](https://img.shields.io/docker/pulls/youngking0727/openbiomed_server)](https://hub.docker.com/repository/docker/youngking0727/openbiomed_server)

![platform](images/platform.png)

Feel free to use our **Agent Platform for Biomedicine and Life Science** at this [website](http://openbiomed.pharmolix.com)!

## News 🎉

- [2026/03/20] 🔥 We release **OpenBioMed Skills**, a comprehensive collection of 45 skills for biomedical research and drug discovery empowered by [Claude Code](https://github.com/anthropics/claude-code). 

> OpenBioMed Skills is a comprehensive skill set released jointly by PharMolix and Institute of AI Industry Research (AIR), Tsinghua University. It provides users with end-to-end solutions for complicated biomedical research tasks spanning drug discovery, protein analysis & engineering, and single-cell omics data analysis. It also presents a copilot mode for creating your own skills by interacting with an LLM agent and the OpenBioMed toolkits. Feel free to have [a quick try](#quick-start-with-claude-code) and [investigate our skills](./skills/skills_overview.md).

- [2026/02/02] 🔥 BioMedGPT-Mol (🤗[HuggingFace Model](https://huggingface.co/PharMolix/BioMedGPT-Mol)) has been released.

> BioMedGPT-Mol is a multimodal molecular language model jointly released by PharMolix Inc. and the Institute of AI Industry Research (AIR), Tsinghua University. It is built for both molecular understanding and generation, supporting a wide range of tasks including chemical name conversion, molecular captioning, property prediction, reaction modeling, molecule editing, and property optimization. Trained with a well-structured multi-task curriculum, BioMedGPT-Mol shows strong performance across diverse molecule-centric discovery benchmarks.

<details>
<summary>Release History</summary>

- [2025/05/26] Our framework has been updated with several new features including new tools, datasets, and models. We implement **LangCell** (📃[Paper](https://arxiv.org/abs/2405.06708), 🤖[Model](https://drive.google.com/drive/folders/1cuhVG9v0YoAnjW-t_WMpQQguajumCBTp?usp=sharing), 📎[Citation](#to-cite-langcell)) and APIs to manipulate cells (See [the Example](./examples/cell_annotation.ipynb)). We also introduce a wider range of tools to calculate molecular properties (ADMET, QED, SA, LogP, Lipinski, Similarity, etc.). 

- [2025/03/07] We present **OpenBioMed Agent Platform** at this [website](http://openbiomed.pharmolix.com) to customize workflows and LLM agents (**AutoPilots**) in solving complicated scientific research tasks. **Tutorials** for using this platform are also [available](https://www.zybuluo.com/icycookies/note/2587490).
- [2025/03/07] Released **OpenBioMed v2**. We present new features including additional downstream biomedical tasks, more flexible data APIs, and advanced models. We also release a preview version of **PharmolixFM** (📃[Paper](https://arxiv.org/abs/2503.21788), 🤖[Model](https://cloud.tsinghua.edu.cn/f/8f337ed5b58f45138659/), 📎[Citation](#to-cite-pharmolixfm)). BioMedGPT-R1 inference is currently supported, and fine-tuning will be available in this month!

> PharmolixFM is an all-atom molecular foundation model jointly released by PharMolix Inc. and Institute of AI Industry Research (AIR), Tsinghua University. It unifies molecules, antibodies, and proteins by jointly modeling them at atom-level with cutting-edge non-autoregressive multi-modal generative models. PharmolixFM is capable of solving mutiple downstream tasks such as docking, structure-based drug design, peptide design, and molecular conformation generation. PharmolixFM achieves competitive performance with AlphaFold3 (83.9 vs 90.2, RMSD < 2Å) on protein-molecule docking (given pocket).


- [2025/02/20] BioMedGPT-R1 (🤗[Huggingface Model](https://huggingface.co/PharMolix/BioMedGPT-R1)) has been released. 

> BioMedGPT-R1-17B is a multimodal biomedical reasoning model jointly released by PharMolix and Institute of AI Industry Research (AIR) . It updates the language model in last version with DeepSeek-R1-Distill-Qwen-14B and adopts two-stage training for cross-modal alignment and multimodal reasoning SFT, performing on par with commercial model on biomedical QA benchmark.

- [2024/05/16] Released implementation of **LangCell** (📃[Paper](https://arxiv.org/abs/2405.06708), 💻[Code](https://github.com/PharMolix/LangCell), 🤖[Model](https://drive.google.com/drive/folders/1cuhVG9v0YoAnjW-t_WMpQQguajumCBTp?usp=sharing), 📎[Citation](#to-cite-langcell)).

> LangCell is the first "language-cell" multimodal pre-trained model jointly developed by PharMolix and Institute for AI Industry Research (AIR). It effectively enhances the understanding of single-cell transcriptomics by learning knowledge-rich texts containing cell identity information, and addresses the task of cell identity understanding in data-scarce scenarios. LangCell is the only single-cell model capable of effective zero-shot cell identity understanding and has also achieved SOTA in few-shot and fine-tuning scenarios.


- [2023/08/14] Released implementation of **BioMedGPT-10B** (📃[Paper](https://arxiv.org/abs/2308.09442v2), 🤖[Model](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F), 📎[Citation](#to-cite-biomedgpt)), **BioMedGPT-LM-7B** (🤗[HuggingFace Model](https://huggingface.co/PharMolix/BioMedGPT-LM-7B)) and **DrugFM** (🤖[Model](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F)).

> BioMedGPT-10B is the first commercial-friendly multimodal biomedical foundation model jointly released by PharMolix and Institute of AI Industry Research (AIR). It aligns the language of life (molecular structures and protein sequences) with human natural language, performing on par with human experts on biomedical QA benchmarks, and demonstrating powerful performance in cross-modal molecule and protein question answering tasks. BioMedGPT-LM-7B is the first commercial-friendly generative foundation model tailored for biomedicine based on Llama-2. 

> DrugFM is a multi-modal molecular foundation model jointly developed by Institute of AI Industry Research (AIR) and Beijing Academy of Artificial Intelligence, BAAI. It leverages UniMAP, a pre-trained molecular model that captures fine-grained properties and representations of molecules, and incorporates MolFM, our multimodal molecular foundation model. DrugFM achieves SOTA on cross-modal retrieval.


- [2023/06/12] Released implementation of **MolFM** (📃[Paper](https://arxiv.org/abs/2307.09484), 🤖[Model](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F), 📎[Citation](#to-cite-molfm)) and **CellLM** (📃[Paper](https://arxiv.org/abs/2306.04371), 🤖[Model](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg), 📎[Citation](#to-cite-celllm)). 

> MolFM is a multi-modal molecular foundation model that enables joint comprehension of molecular structures, biomedical documents and knowledge graphs. On cross-modal retrieval, MolFM outperforms existing models by 12.03% and 5.04% under zero-shot and fine-tuning settings. MolFM also excels in molecule captioning, text-to-molecule generation and molecule property prediction.

> CellLM is the first large-scale cell representation learning model trained on both normal cells and cancer cells with divide-and-conquer contrastive learning. CellLM beats ScBERT on cell type annotation (71.8 vs 68.8), few-shot single-cell drug sensitivity prediction (88.9 vs 80.6) and single-omics cell line drug sensitivity prediction (93.4 vs 87.2).

- [2023/04/23] Released implementation of **BioMedGPT-1.6B** (🤖[Model](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg)) and **OpenBioMed**.

</details>

## Table of contents


- [Introduction](#introduction)
- [Installation](#installation)
- [Quick Start with Claude Code](#quick-start-with-claude-code)
- [Tutorials](#tutorials)
- [Previous version](#previous-version)
- [Limitations](#limitations)
- [Cite us](#cite-us)


## Introduction


This repository holds OpenBioMed, a comprehensive skillset and deep learning toolkit for biomedical discovery. The goal of OpenBioMed is to help researchers build and use **AI-driven workflows for solving complicated biomedical research tasks**. Enpowered by Claude Code, OpenBioMed provides **45 skills** that provides end-to-end solutions for complicated biomedical research tasks. OpenBioMed builds **20+ tools that covers a wide range of downstream applications**, facilitating the construction of your own skills with a seamless user-agent interactions. 


OpenBioMed provide researchers with access to:

- **45 skills** that provides end-to-end solutions for complicated biomedical research tasks, spanning drug discovery, protein analysis & engineering, single-cell omics data analysis, and data retrieval & knowledge.
- **4 types of data modalities**:  OpenBioMed provide easy-to-use APIs for researchers to access and process different types of data including molecules, proteins, pockets, and texts.
- **20+ tools powered by deep learning models**, comprising exclusive models such as [PharmolixFM](https://cloud.tsinghua.edu.cn/f/8f337ed5b58f45138659/), [BioMedGPT-R1](https://huggingface.co/PharMolix/BioMedGPT-R1), [BioMedGPT](https://ieeexplore.ieee.org/document/10767279/) and [MutaPLM](https://arxiv.org/abs/2410.22949).


Here is a list of currently available skills. This is a continuing effort and we are working on further growing the skillset.

#### Biochemistry & Drug Discovery
<details>
<summary>Click here to expand the skills list</summary>

| Skill | Description | Status |
|---|---|---|
| `drug-candidate-discovery` | Generate diverse druggable molecules for a given target or disease using AI-powered drug discovery tools including target identification, structure retrieval, and molecule generation. | MVP |
| `drug-lead-analysis` | Analyze drug candidate molecules for drug-likeness (QED, Lipinski), ADMET properties, BBB penetration, and safety profiles. | MVP |
| `target-based-lead-design` | Generate diverse lead compounds for a specific protein target using structure-based drug design with MolCraft. Includes docking, ADMET filtering, diversity selection, and iterative refinement. | Refined |
| `admet-prediction` | Predict comprehensive ADMET properties (BBB penetration, side effects, Caco-2 permeability, half-life, LD50 toxicity) for drug candidates using GraphMVP ensemble models. | MVP |
| `retrosynthesis-planning` | Expert-in-the-loop retrosynthetic planning workflow for breaking down target molecules into available starting materials and designing synthetic routes with AiZynthFinder integration. | Refined |
| `iupac-name-identification-biot5` | Identify the IUPAC name of a molecule using BioT5 question answering model. | MVP |
| `molecule-biochemical-significance-query-biot5` | Query a molecule's biochemical significance and roles in biology and chemistry using BioT5 multi-modal model. | MVP |
| `text-based-molecule-editing` | Modify molecules based on natural language descriptions using MolT5/BioT5 models for property optimization (solubility, potency, drug-likeness). | MVP |
| `target-drug-report` | Generate comprehensive drug development progress reports for disease therapeutic targets with 7 analysis sections including clinical pipeline, research trends, and patent landscape. | Refined |
| `disease-drug-intelligence` | Analyze disease-to-innovative-drug relationships by querying multiple databases (ChEMBL, ClinicalTrials) to generate comprehensive Chinese reports on disease-target-drug pipelines. | MVP |
| `drug-drug-interaction-analysis` | Analyze potential drug-drug interactions (DDI) for up to 5 drugs using KEGG DDI database with severity levels and mechanism analysis. | MVP |
</details>

#### Protein Analysis & Engineering
<details>
<summary>Click here to expand the skills list</summary>

| Skill | Description | Status |
|---|---|---|
| `protein-mutation-analysis` | Analyze protein mutations by retrieving protein data, explaining mutation effects with MutaPLM, predicting structure with ESMFold, and visualizing results. | Refined |
| `mutation-design-aav` | Design high-fitness and high-diversity mutants of AAV VP1 capsid protein through multi-round iterative optimization. | MVP |
| `mutation-design-gfp` | Design high-fluorescence and high-diversity GFP mutants through multi-round iterative optimization. | MVP |
| `functional-protein-design` | Generate functional protein sequences using CodeFP with Gene Ontology (GO) tag guidance for de novo protein design. | Refined |
| `protein-function-prediction` | Predict protein function and properties from amino acid sequences using BioT5 for functional annotation and pathway analysis. | MVP |
| `similar-protein-retrieval` | Retrieve proteins with similar structures (FoldSeek) or sequences (MSA) from UniProt, PDB, and AFDB databases. | MVP |
| `structure-prediction-boltz-2` | Predict protein complex structures and protein-ligand complexes with binding affinity (IC50) using Boltz-2. | MVP |
| `protein-structure-design-boltzgen` | All-atom protein design using BoltzGen diffusion model for binder design, peptide design, and small molecule binding design. | MVP |
| `antibody-structure-prediction-tfold` | Predict antibody/nanobody structures and antigen-antibody complex structures using tFold model. | MVP |
| `antibody-design-iggm` | Epitope-conditioned de novo antibody design and affinity maturation using IgGM model. | MVP |
| `binding-affinity-prediction-prodigy` | Predict binding affinity scores for protein complexes using Prodigy from structure files. | MVP |
| `protein-ligand-binding-analysis-plip` | Analyze protein-ligand interactions in PDB structures using PLIP for hydrogen bonds, hydrophobic contacts, π-stacking, salt bridges, and visualization. | MVP |
| `protein-subcellular-localization-prediction-biot5` | Predict protein subcellular localization (nucleus, cytoplasm, membrane, etc.) from amino acid sequences using BioT5 model. | MVP |
</details>

#### Single-Cell Omics Data Analysis
<details>
<summary>Click here to expand the skills list</summary>

| Skill | Description | Status |
|---|---|---|
| `single-cell-foundation-model-scrna-seq-geneformer` | Geneformer workflows for tokenization, cell/gene classification, embedding extraction, and in silico perturbation analysis. | MVP |
| `single-cell-foundation-model-scrna-seq-langcell` | LangCell for zero-shot and few-shot cell type annotation with multimodal cell-text matching. | MVP |
| `single-cell-foundation-model-scrna-seq-scgpt` | scGPT for preprocessing, binning, cell embedding extraction, fine-tuning, and reference mapping workflows. | MVP |
| `spatial-transcriptomics-foundation-model-stofm` | SToFM for spatial transcriptomics preprocessing, cell embedding generation with SE(2) Transformer, and downstream analysis. | MVP |
| `single-cell-scrna-seq-analysis-scanpy` | Complete scRNA-seq analysis workflow with Scanpy including QC, normalization, dimensionality reduction, clustering, and marker gene identification. | MVP |
| `single-cell-multi-omics-analysis-scvi` | Probabilistic deep learning for single-cell multi-omics analysis including scVI, scANVI, totalVI, and spatial deconvolution. | MVP |
| `cellxgene-census-query` | Query CZ CELLxGENE Census (61M+ cells) for single-cell expression data by cell type, tissue, or disease. | MVP |
| `spatial-transcriptomics-spatial-data-io` | Load spatial transcriptomics data from Visium, Xenium, MERFISH, Slide-seq, and other platforms using Squidpy and SpatialData. | MVP |
| `single-cell-atac-seq-qc-processing` | Trim adapters, align reads, remove duplicates and mitochondrial contamination, and evaluate chromatin accessibility data quality, including TSS enrichment scoring and fragment size analysis. | MVP |
| `single-cell-atac-seq-peak-calling-annotaion` | Call accessible chromatin peaks with MACS2, annotate peaks to genomic features and genes, and identify differentially accessible regions (DARs) between conditions. | MVP |
| `single-cell-proteomics-data-processing` | Load, inspect, centroid, and extract features from raw LC-MS/MS data files using pyOpenMS, including TIC plotting, feature detection, and format conversion. | MVP |
| `single-cell-proteomics-peptide-identification` | Search MS2 spectra against protein databases with MSFragger/Comet, apply target-decoy FDR filtering, and perform protein inference with parsimony principle. | MVP |
| `single-cell-multi-omics-data-harmonization` | Prepare multi-omics datasets (RNA-seq, proteomics, methylation) for joint integration with per-assay normalization, batch correction, feature ID alignment, and missing value handling. | MVP |
</details>

#### Data Retrieval & Knowledge
<details>
<summary>Click here to expand the skills list</summary>

| Skill | Description | Status |
|---|---|---|
| `pubchem-query` | Query PubChem database for chemical structures, similar compounds (similarity search), and bioactivity data against protein targets. | MVP |
| `uniprot-query` | Query UniProt database for protein sequences, comprehensive metadata (function, domains, diseases), and search by gene name, organism, or keywords. | MVP |
| `chembl-query` | Query ChEMBL database for bioactivity data on drug-like compounds by target, molecule, or disease indication. | MVP |
| `kegg-query` | Query KEGG database for drug information, pathway analysis, and disease-drug-target discovery. | MVP |
| `ppi-string-query` | Query STRING database for protein-protein interactions with confidence scores for network analysis. | MVP |
| `biomedical-literature-search` | Search PubMed and bioRxiv for biomedical research papers with titles, abstracts, and metadata. | MVP |
</details>

#### Utilities
<details>
<summary>Click here to expand the skills list</summary>

| Skill | Description | Status |
|---|---|---|
| `biomed-skill-router` | Find the most suitable skill for a given biomedical task by analyzing user requests and matching against available skill capabilities. | MVP |
| `biomed-skill-creator` | Create new biomedical skills or improve existing ones through an interactive validation process with intent capture, workflow design, and evaluation. | Refined |
</details>

If you are interested in the tools that OpenBioMed skills are built on, please check out the following list.
<details>
<summary>Click here to expand the tools list</summary>

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
|  Structure-based Drug Design   | [PharmolixFM](https://cloud.tsinghua.edu.cn/f/8f337ed5b58f45138659/), [MolCRAFT](https://github.com/AlgoMole/MolCRAFT) | Generate a molecule that binds with a given pocket in a protein |
|     Complex Visualization      |                             N/A                              |             Visualize a protein-molecule complex             |
|      Pocket Visualization      |                             N/A                              |             Visualize a pocket within a protein              |
|          Web Request           |                             N/A                              |             Obtaining information by web search              |
</details>

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

# For PoseBusters
pip install posebusters==0.3.1

# For overlap-based evaluation
pip install spacy rouge_score nltk
python
>>> import nltk
>>> nltk.download('wordnet')
>>> nltk.download('omw-1.4')

# For LangCell
pip install geneformer
```

After downloading the dependencies, you can run the following command to install the package and use our APIs more conveniently:

```bash
pip install -e .
# Try using OpenBioMed APIs
python
>>> from open_biomed.data import Molecule
>>> molecule = Molecule(smiles="CC(=O)OC1=CC=CC=C1C(=O)O")
>>> print(molecule.calc_logp())
```

### Build Docker

Executing ./scripts/docker_run.sh directly will build the Docker image and run the container, launching the backend services on ports 8082 and 8083.
```
sh ./scripts/docker_run.sh
```
At the same time, we also provide a pre-built [docker image](https://hub.docker.com/repository/docker/youngking0727/openbiomed_server), which can be pulled and used directly.    

## Quick Start with Claude Code

OpenBioMed Skills requires [Claude Code](https://github.com/anthropics/claude-code) to be installed and running. 
```bash
mkdir .claude
# Install to your workspace skills directory
cp -r skills/* <your-workspace>/skills/
claude
```

- Type /target-based-lead-design: Configure the target protein or disease (e.g. EGFR) and the desired properties of the lead molecule and receive a bunch of diverse lead candidates with a comprehensive report and visualization after a coffee break!
- Type /functional-protein-design: Give your desired functions (e.g. bacteria degradation), let the model generate a functional protein sequence and its 3D structure.
- Type /biomed-skill-creator: Condense and streamline your workflow into a skill by chatting with an LLM agent.

## Tutorials

Checkout our [Jupytor notebooks](./examples/) for more tutorials!

| Name                                                         | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [BioMedGPT Inference](./examples/biomedgpt_r1.ipynb)         | Examples of using BioMedGPT-10B to answer questions about molecules and proteins and BioMedGPT-R1 to perform reasoning. |
| [Molecule Processing](./examples/manipulate_molecules.ipynb) | Examples of using OpenBioMed APIs to load, process, and export molecules and proteins. |
| [ML Tool Usage](./examples/explore_ai4s_tools.ipynb)         | Examples of using machine learning tools to perform inference. |
| [Visualization](./examples/visualization.ipynb)              | Examples of using OpenBioMed APIs to visualize molecules, proteins, complexes, and pockets. |
| [Workflow Construction](./examples/workflow.ipynb)           | Examples of building and executing workflows and developing LLM agents for complicated scientific tasks. |
| [Model Customization](./examples/model_customization.ipynb)  | Tutorials on how to customize your own model and data using OpenBioMed training pipelines. |

## Other Versions

If you hope to use the features of the previous version, please switch to the `v1.0` branch of this repository by running the following command:

```bash
git checkout v1.0
```

We have also provided a nightly version of OpenBioMed with MCP support. You can try it by running the following command:
```bash
git checkout mcp
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
      author={Luo, Yizhen and Yang, Kai and Fan, Siqi and Hong, Massimo and Zhao, Suyuan and Chen, Xinrui and Nie, Zikun and Luo, Wen and Xie, Ailin and Liu, Xing Yi and Zhang, Jiahuan and Wu, Yushuai and Nie, Zaiqing},
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

```
@article{luo2025pharmolixfm,
  title={PharMolixFM: All-Atom Foundation Models for Molecular Modeling and Generation},
  author={Luo, Yizhen and Wang, Jiashuo and Fan, Siqi and Nie, Zaiqing},
  journal={arXiv preprint arXiv:2503.21788},
  year={2025}
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
