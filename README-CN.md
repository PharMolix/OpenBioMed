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

> OpenBioMed Skills 是由水木分子与清华大学智能产业研究院（AIR）联合发布的一套完整技能集合，为复杂生物医学研究任务提供端到端解决方案，覆盖药物发现、蛋白质分析与设计、单细胞组学数据分析等生物医药热门研究领域。同时，我们提供了 copilot 模式，允许用户通过与 LLM 智能体和 OpenBioMed 工具包交互来创建你自己的技能。欢迎 [快速试用](#claude-code-快速开始) 并 [了解我们的技能](./skills/skills_overview.md)。

- [2026/02/02] 🔥 发布 **BioMedGPT-Mol**（🤗[HuggingFace 模型](https://huggingface.co/PharMolix/BioMedGPT-Mol)）。

> BioMedGPT-Mol 由水木分子与清华大学智能产业研究院（AIR）联合发布的多模态分子语言模型，面向分子理解与生成，支持化学名称转换、分子描述、性质预测、反应建模、分子编辑与性质优化等任务。通过多任务课程训练，在多种分子中心发现基准上表现优异。

<details>
<summary>发布历史</summary>

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

</details>

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

- **45 项技能**：提供复杂生物医学研究任务的端到端解决方案，覆盖药物发现、蛋白质分析与工程、单细胞组学数据分析以及数据检索与知识。
- **4 类数据模态**：分子、蛋白质、口袋与文本的易用访问与处理接口。
- **20+ 工具（由深度学习模型驱动）**：包含如 [PharmolixFM](https://cloud.tsinghua.edu.cn/f/8f337ed5b58f45138659/)、[BioMedGPT-R1](https://huggingface.co/PharMolix/BioMedGPT-R1)、[BioMedGPT](https://ieeexplore.ieee.org/document/10767279/) 与 [MutaPLM](https://arxiv.org/abs/2410.22949) 等自研模型。

以下为当前支持的技能列表（持续更新）：

<details>
<summary>💊 <b>生物化学与药物发现</b>: 先导化合物设计, ADMET预测, 逆合成规划, 分子编辑, 疾病药物情报, 药物-药物相互作用分析. </summary>

| Skill | Description | Status |
|---|---|---|
| `drug-candidate-discovery` | 针对指定靶点或疾病生成多样的可药性候选分子，使用包含靶点识别、结构检索与分子生成的 AI 药物发现工具。 | MVP |
| `drug-lead-analysis` | 对药物候选分子进行药物相似性评估（QED、Lipinski）、ADMET 性质、血脑屏障穿透性与安全性画像分析。 | MVP |
| `target-based-lead-design` | 针对特定蛋白靶点生成多样先导化合物，基于 MolCraft 的基于结构药物设计流程（对接、ADMET 筛选、多样性选择与迭代优化）。 | 高质量 |
| `admet-prediction` | 基于 GraphMVP 集成模型预测候选药物的全面 ADMET 性质（血脑屏障穿透性、副作用、Caco-2 穿透性、半衰期、LD50 毒性）。 | MVP |
| `retrosynthesis-planning` | 面向专家辅助的逆合成规划工作流：将目标分子拆分为可获得的起始原料，并通过 AiZynthFinder 集成设计合成路线。 | 高质量 |
| `iupac-name-identification-biot5` | 使用 BioT5 问答模型识别分子的 IUPAC 命名。 | MVP |
| `molecule-biochemical-significance-query-biot5` | 基于 BioT5 多模态模型查询分子在生物与化学中的生物化学意义与作用。 | MVP |
| `text-based-molecule-editing` | 使用 MolT5/BioT5 模型根据自然语言描述修改分子，并进行性质优化（溶解度、效力、药物相似性）。 | MVP |
| `target-drug-report` | 针对疾病治疗靶点生成完整的药物研发进展报告（包含 7 个分析部分，如临床管线、研究趋势与专利格局）。 | 高质量 |
| `disease-drug-intelligence` | 通过查询多个数据库（ChEMBL、ClinicalTrials）分析疾病与创新药之间的关系，并生成关于疾病-靶点-药物管线的综合中文报告。 | MVP |
| `drug-drug-interaction-analysis` | 通过 KEGG DDI 数据库分析最多 5 种药物的潜在药物-药物相互作用（DDI），并给出严重性与作用机制分析。 | MVP |
</details>

<details>
<summary>🧬 <b>蛋白质分析与设计</b>: 突变分析, 蛋白质从头设计, 结构预测, 结合与相互作用预测, 蛋白质亚细胞定位. </summary>

| Skill | Description | Status |
|---|---|---|
| `protein-mutation-analysis` | 通过检索蛋白数据，对突变效应使用 MutaPLM 进行解释、使用 ESMFold 预测结构并进行可视化展示。 | 高质量 |
| `mutation-design-aav` | 通过多轮迭代优化设计高适配性与高多样性的 AAV VP1 胶囊蛋白突变体。 | MVP |
| `mutation-design-gfp` | 通过多轮迭代优化设计高荧光与高多样性的 GFP 突变体。 | MVP |
| `functional-protein-design` | 借助 CodeFP 并结合基因本体（GO）标签引导进行去 novo 功能蛋白序列设计。 | 高质量 |
| `protein-function-prediction` | 使用 BioT5 从氨基酸序列预测蛋白功能与性质，并进行功能注释与通路分析。 | MVP |
| `similar-protein-retrieval` | 从 UniProt、PDB 与 AFDB 中基于相似结构（FoldSeek）或相似序列（MSA）检索蛋白。 | MVP |
| `structure-prediction-boltz-2` | 使用 Boltz-2 预测蛋白复合物结构以及蛋白-配体复合物，并给出结合亲和力（IC50）。 | MVP |
| `protein-structure-design-boltzgen` | 借助 BoltzGen 扩散模型进行全原子蛋白设计，用于 binder 设计、肽设计与小分子结合设计。 | MVP |
| `antibody-structure-prediction-tfold` | 使用 tFold 预测抗体/纳米抗体结构及抗原-抗体复合物结构。 | MVP |
| `antibody-design-iggm` | 基于 IgGM 进行表位条件的去 novo 抗体设计与亲和力成熟。 | MVP |
| `binding-affinity-prediction-prodigy` | 使用 Prodigy 根据结构文件预测蛋白复合物的结合亲和力评分。 | MVP |
| `protein-ligand-binding-analysis-plip` | 使用 PLIP 在 PDB 结构中分析蛋白-配体相互作用（氢键、疏水接触、π-堆叠、盐桥）并进行可视化。 | MVP |
| `protein-subcellular-localization-prediction-biot5` | 使用 BioT5 根据氨基酸序列预测蛋白亚细胞定位（细胞核、细胞质、膜等）。 | MVP |
</details>

<details>
<summary>🔬 <b>单细胞组学数据分析</b>: RNA组学分析, ATAC组学分析, 多组学数据分析, 空间转录组学分析, 生物信息学分析. </summary>

| Skill | Description | Status |
|---|---|---|
| `single-cell-foundation-model-scrna-seq-geneformer` | 使用 Geneformer 工作流完成 scRNA-seq 的分词、细胞/基因分类、提取与体外扰动分析。 | MVP |
| `single-cell-foundation-model-scrna-seq-langcell` | 使用 LangCell 进行基于细胞-文本多模态匹配的零样本与少样本细胞类型标注。 | MVP |
| `single-cell-foundation-model-scrna-seq-scgpt` | 使用 scGPT 完成 scRNA-seq 预处理、分箱、细胞特征提取、微调与参考映射流程。 | MVP |
| `spatial-transcriptomics-foundation-model-stofm` | 使用 SToFM 完成空间转录组预处理，并利用 SE(2) Transformer 生成细胞特征以支持下游分析。 | MVP |
| `single-cell-scrna-seq-analysis-scanpy` | 使用 Scanpy 构建完整 scRNA-seq 分析流程，包括 QC、归一化、降维、聚类与标记基因识别。 | MVP |
| `single-cell-multi-omics-analysis-scvi` | 使用 scVI/scANVI/totalVI 等概率深度学习方法进行单细胞多组学分析，并支持空间解卷积。 | MVP |
| `cellxgene-census-query` | 查询 CZ CELLxGENE Census（6100 万+ 细胞）按细胞类型、组织或疾病检索单细胞表达数据。 | MVP |
| `spatial-transcriptomics-spatial-data-io` | 使用 Squidpy 和 SpatialData 从 Visium、Xenium、MERFISH、Slide-seq 等平台加载空间转录组数据。 | MVP |
| `single-cell-atac-seq-qc-processing` | 完成 scATAC-seq 的质控处理：去接头、比对、去重与去除线粒体污染，并评估染色质可及性数据质量（包含 TSS 富集评分与片段长度分析）。 | MVP |
| `single-cell-atac-seq-peak-calling-annotaion` | 使用 MACS2 调用可及染色质峰，对峰进行基因组特征与基因注释，并识别不同条件下的差异可及区域（DARs）。 | MVP |
| `single-cell-proteomics-data-processing` | 使用 pyOpenMS 加载、检查、质心化并从原始 LC-MS/MS 数据中提取特征，并包含 TIC 绘图、特征检测与格式转换。 | MVP |
| `single-cell-proteomics-peptide-identification` | 使用 MSFragger/Comet 在蛋白数据库中检索 MS2 谱，进行目标-反目标 FDR 过滤，并基于最大简并原则进行蛋白推断。 | MVP |
| `single-cell-multi-omics-data-harmonization` | 准备多组学数据（RNA-seq、蛋白组学、甲基化）以进行联合整合：支持按实验归一化、批次校正、特征 ID 对齐与缺失值处理。 | MVP |
</details>

<details>
<summary>🔍 <b>数据与知识检索</b>: PubChem, UniProt, ChEMBL, KEGG, STRING, 文献检索. </summary>

| Skill | Description | Status |
|---|---|---|
| `pubchem-query` | 在 PubChem 中检索化学结构、相似化合物（相似性检索）以及针对蛋白靶点的生物活性数据。 | MVP |
| `uniprot-query` | 在 UniProt 中检索蛋白序列与完整元数据（功能、结构域、疾病），并可通过基因名、物种或关键词进行搜索。 | MVP |
| `chembl-query` | 在 ChEMBL 中按靶点/分子/适应症检索药物活性数据。 | MVP |
| `kegg-query` | 查询 KEGG 获取药物信息，支持通路分析以及疾病-药物-靶点发现。 | MVP |
| `ppi-string-query` | 查询 STRING 的蛋白-蛋白相互作用（PPI），基于置信度分数进行网络分析。 | MVP |
| `biomedical-literature-search` | 在 PubMed 与 bioRxiv 中检索生物医学论文，并返回标题、摘要与元数据。 | MVP |
</details>

<details>
<summary>💡 <b>元技能</b>: 技能检索, 技能创建. </summary>

| Skill | Description | Status |
|---|---|---|
| `biomed-skill-router` | 通过分析用户请求并匹配可用能力，为给定生物医学任务寻找最合适的技能。 | MVP |
| `biomed-skill-creator` | 通过与 LLM 智能体的交互式验证流程创建或改进新的生物医学技能（意图捕获、工作流设计与评估）。 | 高质量 |
</details>

以下为当前支持的工具列表：
<details>
<summary>🔧 <b>OpenBioMed 工具</b></summary>

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

</details>

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

## 使用 Claude Code 快速开始

OpenBioMed Skills 需要先安装并运行 [Claude Code](https://github.com/anthropics/claude-code)。

```bash
mkdir .claude
# 安装到你的工作区 skills 目录
cp -r skills/* <your-workspace>/skills/
claude
```

- 输入 `/target-based-lead-design`：配置目标蛋白或疾病（如 EGFR）与先导分子所需性质，稍等片刻即可获得多样先导候选及报告与可视化结果。
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

我们在一个 nightly 分支上提供了 MCP 支持，可以尝试运行以下命令：
```bash
git checkout mcp
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
