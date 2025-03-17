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

欢迎用户在[该网站](http://openbiomed.pharmolix.com)使用我们的生物医药智能体开发平台！

## 更新信息 🎉

- [2025/03/07] 🔥 发布**OpenBioMed生物医药智能体开发平台**，可通过[该链接](http://openbiomed.pharmolix.com)访问。该平台能帮助研发人员零门槛使用AI模型定制化自己的科学研究助手（**AutoPilot**）。平台的[使用文档](https://www.zybuluo.com/icycookies/note/2587490)已经同步发布。

- [2025/03/07] 🔥 发布**OpenBioMed v2**. 我们在这次更新中适配了更多的生物医药下游任务，开放了更加易用的数据接口，并继承了更前沿的AI模型。同时，我们发布了试用版**PharmolixFM**模型（📃[技术报告](), 🤖[模型](https://cloud.tsinghua.edu.cn/f/8f337ed5b58f45138659/), 📎[引用](#to-cite-pharmolixfm)），并完成了BioMedGPT-R1模型的推理支持。我们预计于本月内开放BioMedGPT-R1的微调代码。

    > PharmolixFM是由水木分子与清华大学智能产业研究院联合研发的全原子基础大模型。该模型使用最先进的非自回归式多模态生成模型，在原子尺度上实现了对分子、抗体和蛋白质的联合建模。PharmolixFM能够适配多种下游任务，如分子对接、基于口袋的分子设计、抗体设计、分子构象生成等。在给定口袋位置的分子对接任务中，PharMolixFM的预测精度可与AlphaFold3媲美 (83.9 vs 90.2, RMSD < 2Å) 。

- [2025/02/20] 🔥 发布**BioMedGPT-R1** (🤗[Huggingface模型](https://huggingface.co/PharMolix/BioMedGPT-R1)).
  
    > BioMedGPT-R1-17B是由水木分子与清华大学智能产业研究院（AIR）联合发布的生物医药多模态推理模型。其在上一版本的基础上，用DeepSeek-R1-Distill-Qwen-14B更新了原采用的文本基座模型，并通过跨模态对齐和多模态推理SFT实现模型微调，在生物医药问答任务上效果逼近闭源商用大模型和人类专家水平。

- [2024/05/16]  发布 **LangCell** (📃[论文](https://arxiv.org/abs/2405.06708), 💻[代码](https://github.com/PharMolix/LangCell), 🤖[模型](https://drive.google.com/drive/folders/1cuhVG9v0YoAnjW-t_WMpQQguajumCBTp?usp=sharing), 📎[引用](#to-cite-langcell)).

    > LangCell是由水木分子与清华大学智能产业研究院联合研发的首个“自然语言-单细胞”多模态预训练模型。该模型通过学习富含细胞身份信息的知识性文本，有效提升了对单细胞转录组学的理解能力，并解决了数据匮乏场景下的细胞身份理解任务。LangCell是唯一能有效进行零样本细胞身份理解的单细胞模型，并且在少样本和微调场景下也取得SOTA。LangCell将很快被集成到OpenBioMed。

- [2023/08/14]  发布 **BioMedGPT-LM-7B** (🤗[HuggingFace模型](https://huggingface.co/PharMolix/BioMedGPT-LM-7B)) 、 **BioMedGPT-10B** (📃[论文](https://arxiv.org/abs/2308.09442v2), 🤖[模型](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F), 📎[引用](#to-cite-biomedgpt)) 和 **DrugFM** (🤖[模型](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F)).

    > BioMedGPT-10B是由水木分子联合清华大学智能产业研究院联合发布的首个可商用的多模态生物医药大模型。该模型将以分子结构和蛋白质序列为代表的生命语言与人类的自然语言相结合，在生物医药专业问答能力比肩人类专家水平，在分子和蛋白质跨模态问答中表现出强大的性能。BioMedGPT-LM-7B是首个可商用、生物医药专用的Llama2大模型。

    > DrugFM是由"清华AIR-智源联合研究中心"联合研发的多模态小分子基础模型。 该模型针对小分子药物的组织规律和表示学习进行了更细粒度的设计，形成了小分子药物预训练模型UniMap，并与多模态小分子基础模型MolFM有机结合。该模型在跨模态抽取任务中取得SOTA。

- [2023/06/12] 发布 **MolFM** (📃[论文](https://arxiv.org/abs/2307.09484), 🤖[模型](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F), 📎[引用](#to-cite-molfm)) 和 **CellLM** (📃[论文](https://arxiv.org/abs/2306.04371), 🤖[模型](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg), 📎[引用](#to-cite-celllm)). 

    > MolFM是一个支持统一表示分子结构、生物医学文本和知识图谱的多模态小分子基础模型。在零样本和微调场景下，MolFM的跨模态检索能力分别比现有模型提升了12.03%和5.04%。在分子描述生成、基于文本的分子生成和分子性质预测中，MolFM也取得了显著的结果。

    > CellLM是首个使用分支对比学习策略在正常细胞和癌症细胞数据上同时训练的大规模细胞表示学习模型。CellLM在细胞类型注释（71.8 vs 68.8）、少样本场景下的单细胞药物敏感性预测（88.9 vs 80.6）和单组学细胞系药物敏感性预测上均优于ScBERT（93.4 vs 87.2）。

- [2023/04/23] 发布 **BioMedGPT-1.6B** (🤖[模型](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg)) 和 **OpenBioMed**.

## 目录

- [介绍](#介绍)
- [环境搭建](#环境搭建)
- [使用指南](#使用指南)
- [先前版本](#先前版本)
- [局限性](#局限性)
- [引用](#引用)

## 介绍

OpenBioMed是一个面向生命科学研究和药物研发的Python深度学习工具包。OpenBioMed为小分子结构、蛋白质结构、单细胞转录组学数据、知识图谱和生物医学文本等多模态数据提供了**灵活的数据处理接口**。OpenBioMed构建了**20余个计算工具**，涵盖了大部分AI药物发现任务和最新的针对分子、蛋白质的多模态理解生成任务。此外，OpenBioMed为研究者提供了一套**易用的工作流构建界面**，支持以拖拽形式对接多个模型，并构建基于大语言模型的智能体以解决复杂的科研问题。

OpenBioMed为研究者提供了：

- **4种不同数据的处理接口**， 包括分子结构、蛋白结构、口袋结构和自然语言文本。我们将在未来加入DNA、RNA、单细胞转录组学数据和知识图谱的处理接口。
- **20余个工具**, 包括分子性质预测、蛋白折叠为代表的AI预测工具、分子结构的可视化工具和互联网信息、数据库查询工具。
- **超过20个深度学习模型**, 包括[PharmolixFM](https://cloud.tsinghua.edu.cn/f/8f337ed5b58f45138659/), [BioMedGPT-R1](https://huggingface.co/PharMolix/BioMedGPT-R1), [BioMedGPT](https://ieeexplore.ieee.org/document/10767279/) and [MutaPLM](https://arxiv.org/abs/2410.22949)等自研模型。

OpenBioMed的核心特色如下:

- **统一的数据处理框架**，能轻松加载不同模态的数据，并将其转换为统一的格式。
- **现成的模型预测模块**。我们整理并公开了各类模型的参数，并提供了使用案例，能够简便的迁移到其他数据或任务中。
- **易用的工作流与智能体构建方案**，以帮助研究者针对复杂的科研问题构建多工具协同工作流，通过反复执行工作流以模拟科学试验中的试错过程，并通过大语言模型归纳得到潜在的科学发现。

下表显示了OpenBioMed中支持的工具，它们在未来会被进一步扩展。

|      工具名称       |                           适配模型                           |                             简介                             |
| :-----------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|    分子性质预测     |         [GraphMVP](https://arxiv.org/abs/2110.07728)         |     针对给定分子预测其性质，如血脑屏障穿透性和药物副作用     |
|      分子问答       |          [BioT5](https://arxiv.org/abs/2310.07276)           | 针对给定分子和某个提问进行解答，如介绍分子结构、询问分子官能团、氢键供体的数量等 |
|   分子结构可视化    |                              无                              |                        分子结构可视化                        |
|   分子名称/ID检索   |                              无                              |         基于分子名称或ID，从PubChem数据库中检索分子          |
|  分子相似结构检索   |                              无                              |             从PubChem数据库中检索结构相似的分子              |
|     蛋白质问答      |          [BioT5](https://arxiv.org/abs/2310.07276)           | 针对给定蛋白和某个提问进行解答，如询问motif、蛋白功能、在细胞中的分布和相关疾病等 |
|     蛋白质折叠      | [ESMFold](https://www.science.org/doi/10.1126/science.ade2574) |              基于氨基酸序列预测蛋白质的三维结构              |
|  蛋白结合位点预测   | [P2Rank](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0285-8) |           预测蛋白质中潜在的（与小分子的）结合位点           |
|    突变效应阐释     |         [MutaPLM](https://arxiv.org/abs/2410.22949)          | 给定氨基酸序列上的一个单点突变，使用自然语言描述可能得突变效应 |
|      突变设计       |         [MutaPLM](https://arxiv.org/abs/2410.22949)          | 基于初始蛋白质序列和自然语言描述的优化目标，生成符合优化目标的突变后蛋白质 |
|    蛋白质ID检索     |                              无                              |          基于ID，从UniProtKB数据库中检索蛋白质序列           |
|   蛋白质结构检索    |                              无                              |       基于ID，从PDB和AlphaFoldDB数据库中检索蛋白质结构       |
|  蛋白质结构可视化   |                             N/A                              |                       蛋白质结构可视化                       |
| 蛋白质-分子刚性对接 | [PharmolixFM](https://cloud.tsinghua.edu.cn/f/8f337ed5b58f45138659/) |         给定蛋白口袋结构和分子，生成对接后的分子构象         |
| 基于口袋的分子设计  | [PharmolixFM](https://cloud.tsinghua.edu.cn/f/8f337ed5b58f45138659/) |      给定蛋白口袋结构，生成能与该口袋对接的分子及其构象      |
|    复合物可视化     |                             N/A                              |            可视化蛋白质-小分子结合后的复合物结构             |
|     口袋可视化      |                             N/A                              |                    可视化蛋白质的口袋结构                    |
|     互联网搜索      |                             N/A                              |                      在互联网中检索信息                      |


## 环境搭建

为支持OpenBioMed的基本功能，请执行如下操作：

```bash
conda create -n OpenBioMed python=3.9
conda activate OpenBioMed
pip install torch==1.13.1+{your_cuda_version} torchvision==0.14.1+{your_cuda_version} torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/{your_cuda_version}  
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+{your_cuda_version}.html
pip install pytorch_lightning==2.0.8 peft==0.9.0 accelerate==1.3.0 --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt
```

推荐使用11.7版本的cuda驱动来构建环境。开发者尚未测试使用其他版本的cuda驱动是否会产生问题。

为支持可视化工具与vina分数计算工具，请按如下操作下载依赖包：

```
# 可视化依赖
conda install -c conda-forge pymol-open-source
pip install imageio

# AutoDockVina依赖
pip install meeko==0.1.dev3 pdb2pqr vina==1.2.2 
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3

# 部分评估指标依赖
pip install spacy rouge_score nltk
python
>>> import nltk
>>> nltk.download('wordnet')
>>> nltk.download('omw-1.4')
```
### 构建docker

直接运行 `./scripts/docker_run.sh`，就可以构建docker镜像并运行容器，并在端口8082和8083运行后端服务。
```
sh ./scripts/docker_run.sh
```
与此同时，我们也提供了build好的[docker镜像](https://hub.docker.com/repository/docker/youngking0727/openbiomed_server)，可以直接拉取使用。

## 使用指南

请移步我们的 [使用案例与教程](./examples) 。

| 教程名称                                                     | 简介                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [BioMedGPT推理](./examples/biomedgpt_r1.ipynb)               | 使用BioMedGPT-10B回答分子与蛋白质相关问题和使用BioMedGPT-R1进行推理的示例。 |
| [分子与蛋白质数据处理](./examples/manipulate_molecules.ipynb) | 使用OpenBioMed中的接口加载、处理、导出分子与蛋白质数据的示例。 |
| [深度学习工具的使用](./examples/explore_ai4s_tools.ipynb)    | 使用深度学习模型进行预测的示例。                             |
| [可视化](./examples/visualization.ipynb)                     | 使用OpenBioMed中的接口对小分子、蛋白质、口袋和复合物进行可视化的示例。 |
| [工作流](./examples/workflow.ipynb)                          | 构建多工具协同工作流和大模型智能体的示例。                   |
| [模型开发](./examples/model_customization.ipynb)             | 在OpenBioMed框架中使用个人数据或模型结构开发新模型的教程。   |

## 先前版本

如果你想使用OpenBioMed先前版本的部分功能，请切换至该仓库的v1.0分支：

```bash
git checkout v1.0
```

## 局限性

本项目包含BioMedGPT-LM-7B，BioMedGPT-10B和BioMedGPT-R1，这些模型应当被负责任地使用。BioMedGPT不应用于向公众提供服务。我们严禁使用BioMedGPT生成任何违反适用法律法规的内容，如煽动颠覆国家政权、危害国家安全和利益、传播恐怖主义、极端主义、种族仇恨和歧视、暴力、色情或虚假有害信息等。BioMedGPT不对用户提供或发布的任何内容、数据或信息产生的任何后果负责。

## 协议

本项目代码依照[MIT](./LICENSE)协议开源。使用BioMedGPT-LM-7B、BioMedGPT-10B和BioMedGPT-R1模型，需要遵循[使用协议](./USE_POLICY.md)。

## 联系方式

我们期待您的反馈以帮助我们改进这一框架。若您在使用过程中有任何技术问题或建议，请随时在GitHub issue中提出。若您有商业合作的意向，请联系[opensource@pharmolix.com](mailto:opensource@pharmolix.com)。


## 引用

如果您认为我们的开源代码和模型对您的研究有帮助，请考虑给我们的项目点上星标🌟并引用📎以下文章。感谢您的支持！

##### 引用OpenBioMed:

```
@misc{OpenBioMed_code,
      author={Luo, Yizhen and Yang, Kai and Fan, Siqi and Hong, Massimo and Nie, Zikun and Luo, Wen and Xie, Ailin and Liu, Xing Yi and Zhao, Suyuan and Zhang, Jiahuan and Wu, Yushuai and Nie, Zaiqing},
      title={Code of OpenBioMed},
      year={2023},
      howpublished={\url{https://github.com/Pharmolix/OpenBioMed.git}}
}
```

##### 引用BioMedGPT:

```
@article{luo2024biomedgpt,
  title={Biomedgpt: An open multimodal large language model for biomedicine},
  author={Luo, Yizhen and Zhang, Jiahuan and Fan, Siqi and Yang, Kai and Hong, Massimo and Wu, Yushuai and Qiao, Mu and Nie, Zaiqing},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2024},
  publisher={IEEE}
}
```

##### 引用PharMolixFM:

即将上线

##### 引用MolFM:
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

##### 引用LangCell:
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

##### 引用MutaPLM

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