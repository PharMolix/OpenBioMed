
<div align="center"><h1>OpenBioMed</h1></div>
<h4 align="center">
    <p>
        <b>中文</b> |
        <a href="./README.md">English</a>
    <p>
</h4>

## 更新信息 🎉

- [08/14] 🔥 发布 **BioMedGPT-LM-7B** (🤗[HuggingFace模型](https://huggingface.co/PharMolix/BioMedGPT-LM-7B)) 、 **BioMedGPT-10B** (📃[技术报告](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F), 🤖[模型](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F), 📎[引用](#to-cite-biomedgpt)) 和 **DrugFM** (🤖[模型](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F)).

    > BioMedGPT-10B是由水木分子联合清华大学智能产业研究院联合发布的首个可商用的多模态生物医药大模型。该模型将以分子结构和蛋白质序列为代表的生命语言与人类的自然语言相结合，在生物医药专业问答能力比肩人类专家水平，在分子和蛋白质跨模态问答中表现出强大的性能。BioMedGPT-LM-7B是首个可商用、生物医药专用的Llama2大模型。

    > DrugFM是由"清华AIR-智源联合研究中心"联合研发的多模态小分子基础模型。 该模型针对小分子药物的组织规律和表示学习进行了更细粒度的设计，形成了小分子药物预训练模型UniMap，并与多模态小分子基础模型MolFM有机结合。该模型在跨模态抽取任务中取得SOTA。

- [06/12] 发布 **MolFM** (📃[论文](https://arxiv.org/abs/2307.09484), 🤖[模型](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F), 📎[引用](#to-cite-molfm)) 和 **CellLM** (📃[论文](https://arxiv.org/abs/2306.04371), 🤖[模型](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg), 📎[引用](#to-cite-celllm)). 

    > MolFM是一个支持统一表示分子结构、生物医学文本和知识图谱的多模态小分子基础模型。在零样本和微调场景下，MolFM的跨模态检索能力分别比现有模型提升了12.03%和5.04%。在分子描述生成、基于文本的分子生成和分子性质预测中，MolFM也取得了显著的结果。

    > CellLM是首个使用分支对比学习策略在正常细胞和癌症细胞数据上同时训练的大规模细胞表示学习模型。CellLM在细胞类型注释（71.8 vs 68.8）、少样本场景下的单细胞药物敏感性预测（88.9 vs 80.6）和单组学细胞系药物敏感性预测上均优于ScBERT（93.4 vs 87.2）。

- [04/23] 发布 **BioMedGPT-1.6B** (🤖[模型](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg)) 和 **OpenBioMed**.

## 目录

- [介绍](#介绍)
- [环境搭建](#环境搭建)
- [使用方式](#使用方式)
- [文档](./docs)
- [局限性](#局限性)
- [引用](#引用)

## 介绍

OpenBioMed是一个生物医学的Python深度学习工具包。OpenBioMed提供了多模态生物医学数据的处理接口，包括小分子、蛋白质和单细胞的分子结构、转录组学、知识图谱和生物医学文本数据。OpenBioMed支持广泛的下游应用，包括AI药物发现任务和更具挑战性的多模态理解生成任务。

OpenBioMed为研究者提供了易用的接口，以支持：

- **针对小分子、蛋白质和单细胞的三种模态的数据**， 包括分子结构或转录组学数据、生物医学文本数据和知识图谱。 OpenBioMed为研究人员提供了一个统一的架构来访问、处理和融合多模态数据。
- **10个下游任务**, 包括以药物-靶点亲和力预测、分子性质预测为代表的AI药物研发 (AIDD) 任务，以及以分子描述生成、基于文本的分子生成位代表的多模态任务。
- **超过20个深度学习模型**, 包括 [BioMedGPT-10B](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F), [MolFM](https://arxiv.org/abs/2307.09484), [CellLM](https://arxiv.org/abs/2306.04371) 等。 研究人员可以灵活地组合不同的组件来构建自己的模型。
- **超过20个AI与生物医药交叉领域的热门数据集。** 我们在这些数据集上针对大量模型进行了全面且可复现的评估。

OpenBioMed的核心特色如下:

- **统一的数据处理框架**，能轻松加载不同生物医学实体、不同模态的数据，并将其转换为统一的格式。
- **现成的模型预测模块**。我们公开了预训练的模型的参数，并提供了使用案例，能够简便的迁移到其他数据或任务中。
- **可复现的模型库**，以帮助研究者现有和新的下游任务上快速复现或扩展最先进的模型。

下表显示了OpenBioMed中支持的下游任务与对应的数据集和模型，它们在未来会被进一步扩展。

|                  下游任务                   |         数据集         |              模型               |
| :-------------------------------------: | :--------------------------------: | :-----------------------------------------: |
|          跨模态抽取          |               PCdes                |   KV-PLM, SciBERT, MoMu, GraphMVP, MolFM    |
|           分子描述生成           |              ChEBI-20              |   MolT5, MoMu, GraphMVP, MolFM, BioMedGPT   |
|     基于文本的分子生成      |              ChEBI-20              |         MolT5, SciBERT, MoMu, MolFM         |
|       分子问答       |             ChEMBL-QA              |           MolT5, MolFM, BioMedGPT           |
|       蛋白质问答        |             UniProtQA              |                  BioMedGPT                  |
|        细胞类型注释         |          Zheng68k, Baron           |               scBERT, CellLM                |
|  单细胞药物敏感性预测   |                GDSC                |            DeepCDR, TGSA, CellLM            |
|      分子性质预测       |            MoleculeNet             | MolCLR, GraphMVP, MolFM, DeepEIK, BioMedGPT |
| 药物-靶点亲和力预测 | Yamanishi08, BMKG-DTI, DAVIS, KIBA |         DeepDTA, MGraphDTA, DeepEIK         |
| 蛋白质关系预测  |      SHS27k, SHS148k, STRING       |         PIPR, GNN-PPI, OntoProtein          |

## 环境搭建

1. (可选) 构建conda环境:

```bash
conda create -n OpenBioMed python=3.9
conda activate OpenBioMed
```

2. 下载需要的python包:

```
pip install -r requirements.txt
```

3. 下载Pyg的依赖包:

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-(your_torch_version)+(your_cuda_version).html
pip install torch-geometric
# 如果您在安装上述包时遇到问题，请参阅https://pytorch.org/get-started/locally/和https://github.com/pyg-team/pytorch_geometric。您也可以考虑从https://data.pyg.org/whl/来下载PyTorch Geometric及其依赖。
```

**备注**: 您可能需要下载额外的python包以运行某些下游任务。

## 使用指南

请移步我们的 [使用案例](./examples) 和 [使用文档](./docs)。

| 文件                                                         | 描述                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [使用BioMedGPT-10B进行生成](./examples/biomedgpt_inference.ipynb) | 使用BioMedGPT-10B针对小分子和蛋白质结构进行问答的案例。 |
| [使用MolFM进行跨模态抽取](./examples/cross_modal_retrieval) | 使用MolFM抽取与某个小分子相关性最强的文本的案例。 |
| [使用MolT5进行基于文本的分子生成](./examples/molecule_generation.ipynb) | 使用MolT5，基于文本描述生成小分子的SMILES序列的案例。 |
| [使用CellLM进行细胞类型注释](./examples/cell_type_classification.ipynb) | 使用微调后的CellLM预测细胞类型的案例。   |
| [分子性质预测](./docs/dp.md)                 | 分子性质预测的训练和测试方式 |
| [单细胞药物敏感性预测](./docs/drp.md)                    | 单细胞-药物敏感性预测的训练和测试方式 |
| [药物-靶点亲和力预测](./docs/dti.md)     | 药物-靶点亲和力预测的训练和测试方式 |
| [分子描述生成](./docs/molcap.md)                      | 分子描述生成的训练和测试方式  |

## 局限性

本项目包含BioMedGPT-LM-7B和BioMedGPT-10B，这些模型应当被负责任地使用。BioMedGPT不应用于向公众提供服务。我们严禁使用BioMedGPT生成任何违反适用法律法规的内容，如煽动颠覆国家政权、危害国家安全和利益、传播恐怖主义、极端主义、种族仇恨和歧视、暴力、色情或虚假有害信息等。BioMedGPT不对用户提供或发布的任何内容、数据或信息产生的任何后果负责。

## 协议

本项目代码依照[MIT](./LICENSE)协议开源。使用BioMedGPT-LM-7B和BioMedGPT-10B模型，需要遵循[使用协议](./USE_POLICY.md)。

## 联系方式

我们期待您的反馈以帮助我们改进这一框架。若您在使用过程中有任何技术问题或建议，请随时在GitHub issue中提出。若您有商业合作的意向，请联系[opensource@pharmolix.com](mailto:opensource@pharmolix.com)。


## 引用

如果您认为我们的开源代码和模型对您的研究有帮助，请考虑给我们的项目点一颗星🌟并引用📎以下文章。感谢您的支持！

##### 引用OpenBioMed:

```
@misc{OpenBioMed_code,
      author={Luo, Yizhen and Yang, Kai and Hong, Massimo and Liu, Xing Yi and Zhao, Suyuan and Zhang, Jiahuan and Wu, Yushuai and Nie, Zaiqing},
      title={Code of OpenBioMed},
      year={2023},
      howpublished={\url{https://github.com/BioFM/OpenBioMed.git}}
}
```

##### 引用BioMedGPT:

请等待我们在Arxiv上的更新。

##### 引用DeepEIK:

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

##### 引用CellLM:
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

