<div align="center"><h1>OpenBioMed</h1></div>
<h4 align="center">
    <p>
        <b>中文</b> |
        <a href="./README.md">English</a>
    <p>
</h4>

[![GitHub Repo stars](https://img.shields.io/github/stars/PharMolix/OpenBioMed?style=social)](https://github.com/PharMolix/OpenBioMed/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/PharMolix/OpenBioMed)](https://github.com/PharMolix/OpenBioMed/commits/main)
[![GitHub contributors](https://img.shields.io/github/contributors/PharMolix/OpenBioMed?color=orange)](https://github.com/PharMolix/OpenBioMed/graphs/contributors)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/PharMolix/OpenBioMed/pulls)
[![Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/PharMolix)
[![Docker Pulls](https://img.shields.io/docker/pulls/youngking0727/openbiomed_server)](https://hub.docker.com/repository/docker/youngking0727/openbiomed_server)

![platform](images/platform.png)

欢迎在[该网站](http://openbiomed.pharmolix.com)使用我们的**生物医药与生命科学智能体平台**！

## 更新信息 🎉

- [2026/03/20] 🔥 发布 **OpenBioMed Skills**，一套由 [Claude Code](https://github.com/anthropics/claude-code) 驱动的、涵盖 45 项技能的生物医学研究与药物发现技能集。

- [2026/02/02] 🔥 发布 **BioMedGPT-Mol**（🤗[HuggingFace 模型](https://huggingface.co/PharMolix/BioMedGPT-Mol)）。

> BioMedGPT-Mol 由水木分子与清华大学智能产业研究院（AIR）联合发布的多模态分子语言模型，面向分子理解与生成，支持化学名称转换、分子描述、性质预测、反应建模、分子编辑与性质优化等任务。通过多任务课程训练，在多种分子中心发现基准上表现优异。

- [2025/05/26] 框架更新，包含新工具、数据集与模型。我们实现了 **LangCell**（📃[论文](https://arxiv.org/abs/2405.06708)，🤖[模型](https://drive.google.com/drive/folders/1cuhVG9v0YoAnjW-t_WMpQQguajumCBTp?usp=sharing)，📎[引用](#to-cite-langcell)）及细胞数据操作接口（见[示例](./examples/cell_annotation.ipynb)），并新增 ADMET、QED、SA、LogP、Lipinski、相似性等分子性质计算工具。

- [2025/03/07] 在[该网站](http://openbiomed.pharmolix.com)推出 **OpenBioMed 智能体平台**，用于定制工作流与 LLM 智能体（**AutoPilot**）以解决复杂科研任务。平台[使用教程](https://www.zybuluo.com/icycookies/note/2587490)已同步发布。
- [2025/03/07] 发布 **OpenBioMed v2**。新增更多生物医药下游任务、更灵活的数据接口与先进模型，并发布 **PharmolixFM** 预览版（📃[论文](https://arxiv.org/abs/2503.21788)，🤖[模型](https://cloud.tsinghua.edu.cn/f/8f337ed5b58f45138659/)，📎[引用](#to-cite-pharmolixfm)）。当前支持 BioMedGPT-R1 推理，微调将于本月开放。

> PharmolixFM 由水木分子与清华大学智能产业研究院（AIR）联合发布的全原子分子基础模型，在原子级别对分子、抗体与蛋白质进行联合建模。支持对接、基于结构的药物设计、肽设计、分子构象生成等下游任务。在给定口袋的蛋白-分子对接任务上，PharmolixFM 与 AlphaFold3 表现相当（83.9 vs 90.2，RMSD < 2Å）。

- [2025/02/20] 发布 **BioMedGPT-R1**（🤗[Huggingface 模型](https://huggingface.co/PharMolix/BioMedGPT-R1)）。

> BioMedGPT-R1-17B 由水木分子与清华大学智能产业研究院（AIR）联合发布的生物医药多模态推理模型。采用 DeepSeek-R1-Distill-Qwen-14B 更新语言基座，并通过跨模态对齐与多模态推理 SFT 两阶段训练，在生物医药问答基准上达到与商用模型相当的水平。

- [2024/05/16] 发布 **LangCell** 实现（📃[论文](https://arxiv.org/abs/2405.06708)，💻[代码](https://github.com/PharMolix/LangCell)，🤖[模型](https://drive.google.com/drive/folders/1cuhVG9v0YoAnjW-t_WMpQQguajumCBTp?usp=sharing)，📎[引用](#to-cite-langcell)）。

> LangCell 由水木分子与清华大学智能产业研究院联合研发的首个「自然语言-单细胞」多模态预训练模型，通过学习富含细胞身份信息的文本提升单细胞转录组理解，并解决数据稀缺下的细胞身份理解任务。

- [2023/08/14] 发布 **BioMedGPT-10B**（📃[论文](https://arxiv.org/abs/2308.09442v2)，🤖[模型](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F)，📎[引用](#to-cite-biomedgpt)）、**BioMedGPT-LM-7B**（🤗[HuggingFace 模型](https://huggingface.co/PharMolix/BioMedGPT-LM-7B)）与 **DrugFM**（🤖[模型](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F)）。

> BioMedGPT-10B 由水木分子与清华大学智能产业研究院（AIR）联合发布的首个可商用多模态生物医药基础模型，将分子结构、蛋白质序列等生命语言与自然语言对齐，在生物医药问答上比肩人类专家，在分子与蛋白质跨模态问答中表现突出。BioMedGPT-LM-7B 为基于 Llama-2 的首个可商用生物医药生成基础模型。

> DrugFM 由清华大学智能产业研究院（AIR）与北京智源研究院联合研发的多模态分子基础模型，基于 UniMAP 预训练分子表示并融合 MolFM，在跨模态检索上达到 SOTA。

- [2023/06/12] 发布 **MolFM**（📃[论文](https://arxiv.org/abs/2307.09484)，🤖[模型](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F)，📎[引用](#to-cite-molfm)）与 **CellLM**（📃[论文](https://arxiv.org/abs/2306.04371)，🤖[模型](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg)，📎[引用](#to-cite-celllm)）。

> MolFM 为多模态分子基础模型，支持分子结构、生物医学文献与知识图谱的联合理解。在跨模态检索上，零样本与微调设定下分别较现有模型提升 12.03% 与 5.04%，并在分子描述、文本到分子生成与分子性质预测上表现优异。

> CellLM 为首个在正常细胞与癌细胞上采用分治对比学习的大规模细胞表示模型。在细胞类型注释（71.8 vs 68.8）、少样本单细胞药物敏感性预测（88.9 vs 80.6）与单组学细胞系药物敏感性预测（93.4 vs 87.2）上均优于 ScBERT。

- [2023/04/23] 发布 **BioMedGPT-1.6B**（🤖[模型](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg)）与 **OpenBioMed**。

## 目录

- [介绍](#介绍)
- [环境搭建](#环境搭建)
- [Claude Code 快速开始](#claude-code-快速开始)
- [教程](#教程)
- [先前版本](#先前版本)
- [局限性](#局限性)
- [引用](#引用)

## 介绍

本仓库为 OpenBioMed，一个面向 AI 赋能生物医药的 Python 深度学习工具包。OpenBioMed 旨在帮助研究者构建并使用 **AI 驱动的工作流以解决复杂生物医学研究任务**。OpenBioMed 提供 **20+ 工具**，覆盖从传统 AI 药物发现到新兴多模态任务等多种下游应用。在 Claude Code 支持下，OpenBioMed 提供 **45 项技能**，为复杂生物医学研究提供端到端方案，并通过顺畅的人机协作便于你构建自己的技能。

OpenBioMed 为研究者提供：

- **4 类数据模态**：分子、蛋白质、口袋与文本的易用访问与处理接口。
- **20+ 工具**：涵盖 AIDD 中的分子性质预测、蛋白质折叠等 ML 预测模型，以及可视化与网络检索接口。
- **20+ 深度学习模型**：包括 [PharmolixFM](https://cloud.tsinghua.edu.cn/f/8f337ed5b58f45138659/)、[BioMedGPT-R1](https://huggingface.co/PharMolix/BioMedGPT-R1)、[BioMedGPT](https://ieeexplore.ieee.org/document/10767279/) 与 [MutaPLM](https://arxiv.org/abs/2410.22949) 等自研模型。
- **45 项技能**：覆盖药物发现、蛋白质分析与工程、单细胞组学分析与数据检索等复杂生物医学任务的端到端解决方案。

OpenBioMed 的主要特点包括：

- **统一数据处理流程**：便捷加载并将不同生物医学实体与模态的异构数据转换为统一格式。
- **开箱即用推理**：公开的预训练模型与推理示例，可方便迁移至自有数据或任务。
- **易用的复杂工作流构建与使用**：提供基于技能执行复杂工作流的 autopilot 模式，以及通过 LLM 智能体与 OpenBioMed 工具包交互、用于创建自定义技能的 [copilot 模式](./skills/biomed-skill-creator/)。

以下为当前支持的工具列表，我们将持续扩展。

|              工具              |                           支持模型                           |                              描述                              |
| :----------------------------: | :----------------------------------------------------------: | :------------------------------------------------------------: |
| 分子性质预测                    |         [GraphMVP](https://arxiv.org/abs/2110.07728)         | 预测给定分子的性质（如血脑屏障穿透性、副作用等）                 |
| 分子问答                        |          [BioT5](https://arxiv.org/abs/2310.07276)           | 回答关于给定分子的文本查询（如结构描述、官能团、氢键供体数量等） |
| 分子可视化                      |                             N/A                              | 分子可视化                                                     |
| 分子名称/ID 检索                |                             N/A                              | 通过名称或 PubChemID 从 PubChem 获取分子                       |
| 分子结构检索                    |                             N/A                              | 基于相似结构从 PubChem 获取分子                                |
| 蛋白质问答                      |          [BioT5](https://arxiv.org/abs/2310.07276)           | 回答关于给定蛋白质的文本查询（如 motif、功能、亚细胞定位、相关疾病等） |
| 蛋白质折叠                      | [ESMFold](https://www.science.org/doi/10.1126/science.ade2574) | 根据氨基酸序列预测蛋白质三维结构                               |
| 蛋白质口袋预测                  | [P2Rank](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0285-8) | 预测蛋白质内潜在结合位点                                       |
| 突变解释                        |         [MutaPLM](https://arxiv.org/abs/2410.22949)          | 对蛋白质序列上的单点替换突变给出文本解释                         |
| 突变设计                        |         [MutaPLM](https://arxiv.org/abs/2410.22949)          | 根据野生型序列与文本指令生成突变后蛋白质                         |
| 蛋白质 UniProtID 检索           |                             N/A                              | 根据 UniProt 登录号从 UniProtKB 获取蛋白质序列                  |
| 蛋白质 PDB 检索                |                             N/A                              | 根据 PDB/AlphaFoldDB 登录号从 PDB/AlphaFoldDB 获取蛋白质结构    |
| 蛋白质可视化                    |                             N/A                              | 蛋白质可视化                                                   |
| 蛋白质-分子刚性对接             | [PharmolixFM](https://cloud.tsinghua.edu.cn/f/8f337ed5b58f45138659/) | 生成分子与给定蛋白口袋的结合构象                               |
| 基于结构的药物设计              | [PharmolixFM](https://cloud.tsinghua.edu.cn/f/8f337ed5b58f45138659/)、[MolCRAFT](https://github.com/AlgoMole/MolCRAFT) | 生成与给定蛋白口袋结合的分子                                   |
| 复合物可视化                    |                             N/A                              | 蛋白质-分子复合物可视化                                        |
| 口袋可视化                      |                             N/A                              | 蛋白质内口袋可视化                                             |
| 网络检索                        |                             N/A                              | 通过网络搜索获取信息                                           |

## 环境搭建

如需启用 OpenBioMed 的基本功能，请执行：

```bash
conda create -n OpenBioMed python=3.9
conda activate OpenBioMed
pip install torch==1.13.1+{your_cuda_version} torchvision==0.14.1+{your_cuda_version} torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/{your_cuda_version}  
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+{your_cuda_version}.html
pip install pytorch_lightning==2.0.8 peft==0.9.0 accelerate==1.3.0 --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt
```

建议使用 cuda=11.7 配置环境，其他版本的 cudatoolkit 可能导致异常。

若要使用可视化工具与 vina 分数计算工具，请安装以下依赖：

```
# 可视化
conda install -c conda-forge pymol-open-source
pip install imageio

# AutoDockVina
pip install meeko==0.1.dev3 pdb2pqr vina==1.2.2 
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3

# PoseBusters
pip install posebusters==0.3.1

# 基于重叠的评估
pip install spacy rouge_score nltk
python
>>> import nltk
>>> nltk.download('wordnet')
>>> nltk.download('omw-1.4')

# LangCell
pip install geneformer
```

依赖安装完成后，可执行以下命令安装包并更方便地使用接口：

```bash
pip install -e .
# 尝试使用 OpenBioMed 接口
python
>>> from open_biomed.data import Molecule
>>> molecule = Molecule(smiles="CC(=O)OC1=CC=CC=C1C(=O)O")
>>> print(molecule.calc_logp())
```

### 构建 Docker

直接执行 `./scripts/docker_run.sh` 将构建 Docker 镜像并运行容器，在 8082 和 8083 端口启动后端服务。

```
sh ./scripts/docker_run.sh
```

我们也提供预构建的 [Docker 镜像](https://hub.docker.com/repository/docker/youngking0727/openbiomed_server)，可直接拉取使用。

## Claude Code 快速开始

OpenBioMed Skills 需要先安装并运行 [Claude Code](https://github.com/anthropics/claude-code)。

```bash
mkdir .claude
# 安装到你的工作区 skills 目录
cp -r skills/* <your-workspace>/skills/
claude
```

- 输入 `/target-based-lead-design`：配置目标蛋白或疾病（如 EGFR）与先导分子所需性质，稍等片刻即可获得多样先导候选及报告与可视化。
- 输入 `/functional-protein-design`：给出期望功能（如 溶菌酶），由模型生成具有该功能的蛋白质序列及其三维结构。
- 输入 `/biomed-skill-creator`：通过与 LLM 智能体对话，将你的工作流整理并固化为一项技能。

## 教程

更多教程请参见 [Jupyter notebooks](./examples/)。

| 名称                                                           | 描述                                                           |
| -------------------------------------------------------------- | -------------------------------------------------------------- |
| [BioMedGPT 推理](./examples/biomedgpt_r1.ipynb)                | 使用 BioMedGPT-10B 回答分子与蛋白质问题，以及使用 BioMedGPT-R1 进行推理的示例。 |
| [分子处理](./examples/manipulate_molecules.ipynb)              | 使用 OpenBioMed 接口加载、处理与导出分子与蛋白质的示例。       |
| [ML 工具使用](./examples/explore_ai4s_tools.ipynb)             | 使用机器学习工具进行推理的示例。                               |
| [可视化](./examples/visualization.ipynb)                       | 使用 OpenBioMed 接口可视化分子、蛋白质、复合物与口袋的示例。   |
| [工作流构建](./examples/workflow.ipynb)                        | 构建与执行工作流、以及为复杂科研任务开发 LLM 智能体的示例。   |
| [模型定制](./examples/model_customization.ipynb)               | 使用 OpenBioMed 训练流程定制自有模型与数据的教程。             |

## 先前版本

如需使用旧版功能，请切换到本仓库的 `v1.0` 分支：

```bash
git checkout v1.0
```

## 局限性

本仓库包含 BioMedGPT-LM-7B、BioMedGPT-10B 与 BioMedGPT-R1，我们强调对这些模型负责任与合乎伦理的使用。BioMedGPT 不得用于向公众提供服务。严禁生成任何违反适用法律法规的内容，包括但不限于煽动颠覆国家政权、危害国家安全与利益、传播恐怖主义、极端主义、民族仇恨与歧视、暴力、色情或虚假有害信息等。BioMedGPT 不对用户提供或发布的任何内容、数据或信息所导致的后果承担责任。

## 协议

本仓库采用 [MIT License](./LICENSE)。使用 BioMedGPT-LM-7B 与 BioMedGPT-10B 模型须遵守 [可接受使用政策](./USE_POLICY.md)。

## 联系我们

我们欢迎用户反馈以改进框架。如有技术问题或建议，欢迎提交 issue。商业支持或合作请联系 [opensource@pharmolix.com](mailto:opensource@pharmolix.com)。

## 引用

若本开源代码与模型对你的研究有帮助，欢迎为本仓库加星 🌟 并引用 📎 相关论文，感谢支持。

##### 引用 OpenBioMed：

```
@misc{OpenBioMed_code,
      author={Luo, Yizhen and Yang, Kai and Fan, Siqi and Hong, Massimo and Zhao, Suyuan and Chen, Xinrui and Nie, Zikun and Luo, Wen and Xie, Ailin and Liu, Xing Yi and Zhang, Jiahuan and Wu, Yushuai and Nie, Zaiqing},
      title={Code of OpenBioMed},
      year={2023},
      howpublished={\url{https://github.com/Pharmolix/OpenBioMed.git}}
}
```

##### 引用 BioMedGPT：

```
@article{luo2024biomedgpt,
  title={Biomedgpt: An open multimodal large language model for biomedicine},
  author={Luo, Yizhen and Zhang, Jiahuan and Fan, Siqi and Yang, Kai and Hong, Massimo and Wu, Yushuai and Qiao, Mu and Nie, Zaiqing},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2024},
  publisher={IEEE}
}
```

##### 引用 PharmolixFM：

```
@article{luo2025pharmolixfm,
  title={PharMolixFM: All-Atom Foundation Models for Molecular Modeling and Generation},
  author={Luo, Yizhen and Wang, Jiashuo and Fan, Siqi and Nie, Zaiqing},
  journal={arXiv preprint arXiv:2503.21788},
  year={2025}
}
```

##### 引用 MolFM：

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

##### 引用 LangCell：

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

##### 引用 MutaPLM：

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
