# OpenBioMed 平台模块与接口调研

> 目标：按照“大模型平台架构 v0.1”的模块划分，调研 OpenBioMed 仓库中是否存在对应模块，并整理模块之间的箭头/接口关系，包括输入、输出数据类型、工具调用与缺口。

## 1. 总体结论

OpenBioMed 不是一个完整的“任务队列 + 多执行器 + 算法评测器”生产平台实现，而是一个以 **OpenBioMed 数据对象 + Tool 注册表 + Pipeline/Task/Model 推理 + Workflow DAG + Agent PlannerExecutor + FastAPI 服务** 为核心的生物医学大模型工具平台。

与参考架构对齐后可以得到：

| 参考架构模块 | OpenBioMed 对应实现 | 存在程度 | 说明 |
|---|---|---:|---|
| 应用层 | `open_biomed/scripts/run_server.py`、`run_server_workflow.py`、`open_biomed/scripts/chat.py`、`examples/`、`demo_workflows/` | 部分存在 | 后端 API、脚本、示例存在；仓库内没有完整前端应用源码。 |
| 任务规划器 / 智能路由 | `open_biomed/core/agent.py::PlannerExecutor`、`TOOLS`/`WORKFLOWS` 注入 prompt、`parse_frontend()` | 存在 | LLM 根据工具说明规划并执行；前端导出的图可转 YAML workflow。 |
| 任务队列管理器 | `Workflow.exec_queue`、LangGraph state、`asyncio.create_task()` | 弱存在 | 有内存队列/执行队列，不是持久化任务队列，也没有优先级、重试队列、资源调度队列。 |
| 任务调度器 / 管线隔离 | `Workflow` DAG 拓扑执行、`InferencePipeline`、可选 Docker 执行、可视化 subprocess | 部分存在 | Workflow 是 DAG 调度；可视化用子进程规避 PyMol 问题；Agent 可选 Docker，但没有统一集群调度/隔离层。 |
| 内部数据 | `open_biomed/data/*`、`datasets/*`、`memory/workflows/*`、`configs/*` | 存在 | 统一对象包括 Molecule、Protein、Pocket、Text、Cell；训练/评测数据集也有 Registry。 |
| 外部数据 | `web_request_tools.py`：PubChem、UniProt、PDB、STRING、ChEMBL、WebSearch 等 | 存在 | 通过异步 HTTP 查询外部数据库/搜索服务，返回 OpenBioMed 对象或结构化数据。 |
| 靶点发现执行器/算法仓库 | `protein_binding_site_prediction`、`protein_question_answering`、`ppi_string_request`、skills 中靶点相关能力 | 部分存在 | 代码层主要是结合位点/PPI/蛋白问答；没有独立命名的“靶点发现算法仓库”。 |
| 分子生成执行器/算法仓库 | `structure_based_drug_design`、`text_guided_molecule_generation`、`text_based_molecule_editing`、`pocket_molecule_docking` | 存在 | 支持基于口袋生成、文本引导生成/编辑、对接。 |
| 性能预测执行器/算法仓库 | `molecule_property_prediction`、`molecule_qed/sa/logp/lipinski/similarity`、`protein_molecule_docking_score` | 存在 | ML 预测 + 规则/化学指标 + docking score。 |
| 合成规划执行器/算法仓库 | skills 中 `retrosynthesis-planning`，但核心 `TOOLS` 无对应工具 | 弱存在 | Skills 层有逆合成规划；核心 Python tool registry 没有合成路线规划接口。 |
| 临床优化执行器/算法仓库 | `molecule_property_prediction`、`text_based_molecule_editing`、skills 中 ADMET/lead analysis | 部分存在 | 更接近 ADMET/性质优化；没有临床试验/真实世界证据优化执行器。 |
| 算法评测器 | `TrainValPipeline`、Task callbacks/monitor、`protein_molecule_docking_score`、性质计算工具 | 部分存在 | 有训练/测试评测和若干任务指标工具，但没有统一“新算法接入-评测-发布”平台模块。 |

## 2. 核心数据接口

OpenBioMed 的模块间箭头主要不是传裸 JSON，而是传 Python 对象；HTTP 层再把文件路径、SMILES/FASTA、文本转换为对象。

### 2.1 数据对象

| 数据类型 | 构造输入 | 主要字段 | 输出/序列化 | 典型流向 |
|---|---|---|---|---|
| `Molecule` | SMILES、SELFIES、RDKit Mol、PDB/SDF/PDBQT/PKL 文件 | `smiles`、`rdmol`、`graph`、`conformer`、`description`、`kg_accession` | `.sdf`、`.pkl`、字符串 SMILES | 分子问答、性质预测、生成/编辑、对接、可视化、导出 |
| `Protein` | FASTA、PDB/PKL 文件 | `sequence`、`residues`、`all_atom`、`description`、`kg_accession` | `.pdb`、`.pkl`、字符串序列 | 蛋白问答、折叠、突变设计、结合位点预测、可视化 |
| `Pocket` | 蛋白子序列 residue indices、蛋白+参考配体、PDB/PKL 文件 | `atoms`、`conformer`、`orig_indices`、`orig_protein` | `.pdb`、`.pkl` | SBDD、对接、口袋可视化 |
| `Text` | 字符串 | `str` | 字符串 | QA、编辑/优化 prompt、摘要 |
| `Cell` | `scanpy.AnnData` 或序列 | `anndata`、`sequence` | 对象 | 单细胞注释任务 |

### 2.2 统一 Tool 输入转换

`create_tool_input(data_type, value)` 的约定：

| `data_type` | 输入字符串解释 | 输出对象 |
|---|---|---|
| `molecule` | `.sdf` -> `Molecule.from_sdf_file`；`.pkl` -> `Molecule.from_binary_file`；否则按 SMILES | `Molecule` |
| `protein` | `.pdb` -> `Protein.from_pdb_file`；同名 `.pdb` 存在时优先；`.pkl` -> `Protein.from_binary_file`；否则按 FASTA | `Protein` |
| `pocket` | `.pkl` | `Pocket` |
| `text` | 原字符串 | `Text` |
| 其他 | 原值 | `Any` |

### 2.3 Tool 标准返回

所有 Tool 遵循 `run(...) -> Tuple[List[Any], List[Any]]`：

- 第一个列表：真实输出对象，供后续工具传递。
- 第二个列表：观测/文件路径/前端展示文本，供 UI、日志或下载使用。
- `serial_exec` 装饰器支持把 list 输入逐条执行并拼接输出。
- `wrap_outputs()` / `wrap_and_select_outputs()` 根据输出对象类型封装成 `molecule/protein/pocket/text/output`，供 workflow 边自动注入下游输入。

## 3. 模块级接口与箭头

### 3.1 应用层 -> 任务规划器/工具服务

#### HTTP API

| Endpoint | 请求模型 | 关键输入字段 | 调用目标 | 返回 |
|---|---|---|---|---|
| `POST /run_pipeline/` | `TaskRequest` | `task`, `model`, `molecule`, `protein`, `pocket`, `text`, `dataset`, `mutation`, `indices`, `property` 等 | `TASK_CONFIGS` -> `TOOLS[pipeline_key]` -> handler | 任务名 + 模型 + molecule/protein/pocket/text/score/image/file 等 |
| `POST /web_search/` | `SearchRequest` | `task`, `query`, `molecule`, `threshold` | 异步 requester tool | 外部数据查询结果，例如 molecule/protein/text |
| `POST /run_workflow/` | `ReportRequest` | `task`, `workflow`, `user_email`, `num_repeats` | `ReportGeneratorSBDD` 或 `ReportGeneratorGeneral` | 立即返回“Workflow is still running...”，后台异步生成报告 |

#### 脚本/Chat

- `open_biomed/scripts/inference.py` 提供一组 `test_*` 构造函数，把下游任务封装为 `InferencePipeline` 或 `VinaDockTask`，并被 `TOOLS` 懒加载。
- `open_biomed/core/agent.py::PlannerExecutor` 面向自然语言任务：先生成计划，再在 `<execute>` 中执行 Python/Bash，或在 `<report>` 中输出报告。

### 3.2 任务规划器 / 智能路由

#### 实现

- `PlannerExecutor` 初始化：读取 agent 配置，创建 LLM，选择 context manager，初始化工具、memory workflow、system prompt、执行环境和 LangGraph 工作流。
- 工具注入：若未配置 tool retriever，则把 `TOOLS.available_tools()` 中所有工具的 `print_usage()` 拼到 prompt。
- Workflow memory：若开启 `memory.workflow`，把 `WORKFLOWS` 描述、输入、输出注入 prompt，允许 Agent 直接调用预定义 workflow。
- 执行语言：Python 或 Bash；Python 复用持久 namespace，Bash 可选 Docker。

#### 输入

| 输入来源 | 数据结构 |
|---|---|
| 用户自然语言 | `user_prompt: str` |
| Agent 配置 | `timeout`, `plan_style`, `critic`, `tool_call`, `use_docker`, `memory`, `llm`, `context_manager` 等 |
| 工具说明 | `TOOLS[tool].print_usage()` |
| Workflow 说明 | `workflow.metadata` |

#### 输出

| 输出 | 说明 |
|---|---|
| `thread_id` | 会话目录 `tmp/planner_executor-{thread_id}` |
| `messages` | LangGraph/对话上下文 |
| `captured_results` | 保存的 figure/molecule/protein/visualization 文件 |
| `report.md/pdf` | 若 LLM 输出 `<report>`，可导出报告 |

#### 路由逻辑

1. LLM 生成包含 `<execute>` 或 `<report>` 的响应。
2. 若是 `<execute>`，执行 Python/Bash，并把 stdout/stderr 作为 observation 回写。
3. 若是 `<report>`，结束。
4. 连续解析失败或代码失败达到容忍阈值后结束或要求重试。

### 3.3 前端 Workflow 图 -> YAML Workflow

`parse_frontend(json_string)` 把前端节点/边 JSON 转换为内部 YAML：

- 读取 `nodes` 与 `edges`。
- 过滤 `ChatInput`、`ChatOutput`、`ParseData` 等前端节点。
- 提取 tool 节点的 `config/molecule/protein/pocket/text/dataset/query/mutation/indices/threshold` 等输入。
- 合并 `MergeDataComponent`、`ParseData` 边。
- 生成：
  - `tools: [{name, inputs?}, ...]`
  - `edges: [{start, end, name_mapping?}, ...]`
- 对若干前端字段做硬编码映射，例如 `molecule_property_prediction.dataset -> task`、`molecule_name_request.query -> accession`。

### 3.4 任务队列管理器与任务调度器

OpenBioMed 的调度是 **DAG 内存调度**：

| 模块 | 实现 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| Workflow DAG 构建 | `Workflow(config)` | YAML `metadata/tools/edges` | `nodes`, `edges`, `output_nodes` | tool name 解析到 `TOOLS`；`code_execution` 特殊处理 |
| 执行队列 | `exec_queue`, `in_deg` | DAG 入度 | 待执行节点序号 | 按拓扑顺序执行；无持久化/优先级 |
| 边传递 | `wrap_outputs()` + `name_mapping` | 上游 Tool 输出对象列表 | 下游 node.inputs | 默认按类型名注入；可通过 `name_mapping` 改字段名 |
| 重复执行 | `num_repeats` | 节点配置 | 多次输出 append | 节点级重复 |
| 中断 | `deamon.should_interrupt(state)` | runtime state | bool | 可中断，但没有完整外部调度器 |

箭头形式：

```text
Application/API/Agent
  -> Config/YAML Workflow
  -> Workflow(nodes, edges)
  -> exec_queue 按拓扑取 node
  -> TOOLS[node.name].run(**inputs)
  -> wrap_outputs(outputs)
  -> edge.name_mapping 重命名
  -> downstream_node.inputs
  -> output_nodes results/messages
```

### 3.5 数据层：内部数据与外部数据

#### 内部数据

- `open_biomed/data/`：运行时数据对象。
- `open_biomed/datasets/`：训练/评测数据集；通过 `DATASET_REGISTRY` 被 `DefaultDataModule` 载入。
- `configs/`：模型、数据集、可视化、workflow 配置。
- `memory/workflows/`：Agent 可用的预定义 workflow memory。
- `skills/`：更高层的生物医学工作流能力说明与脚本资产。

#### 外部数据工具

| Tool | 外部系统 | 输入 | 输出 |
|---|---|---|---|
| `molecule_name_request` / `pubchemid_search` | PubChem compound by CID/name | `accession`/`query` | `Molecule`, `.pkl` 路径 |
| `molecule_structure_request` | PubChem similarity API | `Molecule`, `threshold`, `max_records` | 相似 `Molecule` 列表 |
| `pubchem_bioactivity` | PubChem PUG View / bioactivity | molecule/name/id | bioactivity dict |
| `protein_uniprot_request` | UniProt | accession | `Protein` 或 FASTA/PKL 路径 |
| `protein_pdb_request` | PDB / AlphaFoldDB | accession, mode | `Protein`、PDB 文件路径或 metadata |
| `ppi_string_request` | STRING | UniProt/gene, species, score, limit | interaction partner dict |
| `chembl_query` | ChEMBL | molecule/target/indication query | activity/phase dict |
| `web_search` | 搜索引擎 | query | 拼接文本结果 |

### 3.6 执行器与算法仓库

OpenBioMed 通过 `TOOLS` 将执行器暴露为统一工具；重模型工具一般是 `InferencePipeline`，轻量/外部工具是 `Tool` 子类。

#### 3.6.1 靶点发现相关

| Tool/能力 | 输入 | 输出 | 底层 |
|---|---|---|---|
| `protein_binding_site_prediction` | `protein: Protein` | `Pocket` 列表 + `.pkl` 路径 | P2Rank CLI (`third_party/p2rank_2.5/prank`) |
| `protein_question_answering` | `protein: Protein`, `text: Text` | `Text` answer | BioT5 pipeline |
| `ppi_string_request` | `uniprot_id`, `species`, `required_score`, `limit` | PPI partners/confidence | STRING API |
| `protein_pdb_request` / `protein_uniprot_request` | accession | Protein/PDB/metadata | PDB/UniProt API |

#### 3.6.2 分子生成 / 设计相关

| Tool/能力 | 输入 | 输出 | 底层模型/算法 |
|---|---|---|---|
| `structure_based_drug_design` | `pocket: Pocket` | `Molecule` with likely pocket binding | MolCRAFT or PharmolixFM checkpoint configuration |
| `text_guided_molecule_generation` | `text: Text` | `Molecule` | task/model registry 支持，未在 server `TOOLS.available_tools()` 中暴露 |
| `text_based_molecule_editing` | `molecule: Molecule`, `text: Text` | edited `Molecule` | MolT5/BioT5 style pipeline |
| `pocket_molecule_docking` | `molecule: Molecule`, `pocket: Pocket` | `Molecule` with 3D binding pose | PharmolixFM pipeline |
| `mutation_engineering` | `protein: Protein`, `text: Text` | mutation strings；server 进一步转 mutated `Protein` | MutaPLM pipeline |
| `go_guided_protein_generation` | GO term list | protein sequence | CodeFP pipeline |

#### 3.6.3 性能预测 / 打分相关

| Tool/能力 | 输入 | 输出 | 底层 |
|---|---|---|---|
| `molecule_property_prediction` | `molecule: Molecule`, `task/dataset` | property score/text | GraphMVP ensemble, BBBP/SIDER/regression 等 |
| `molecule_qed` | `molecule: Molecule` | QED float | RDKit QED |
| `molecule_sa` | `molecule: Molecule` | SA score | RDKit + fpscores |
| `molecule_logp` | `molecule: Molecule` | LogP float | RDKit Crippen |
| `molecule_lipinski` | `molecule: Molecule` | Lipinski count | RDKit descriptors |
| `molecule_similarity` | `molecule_1`, `molecule_2` | fingerprint similarity | RDKit fingerprint |
| `protein_molecule_docking_score` | `molecule: Molecule`, `protein: Protein` | Vina score / pose | VinaDockTask |

#### 3.6.4 合成规划相关

- 核心 Python `TOOLS` 中没有 `retrosynthesis_planning` / synthesis route planning tool。
- `skills/retrosynthesis-planning` 表明仓库在 Skills 层规划了该能力，但不属于当前 core/server 的可直接调用工具。
- 因此“合成规划执行器/算法仓库”在 core 平台中是缺口，在 skills 层是弱存在。

#### 3.6.5 临床优化相关

- 无独立“临床优化执行器”。
- 可组合能力包括：`molecule_property_prediction`（如 SIDER、BBBP、ADMET 类任务）、`text_based_molecule_editing`（文本目标优化）、`protein_molecule_docking_score`（结合评分）、skills 中 `admet-prediction`、`drug-lead-analysis`、`target-drug-report`。
- 更准确的定位：OpenBioMed 当前支持“先导化合物性质优化/ADMET 风险评估”，不支持临床流程调度或临床数据闭环优化。

### 3.7 算法评测器

OpenBioMed 的评测分散在三层：

| 层 | 实现 | 输入 | 输出 | 适用范围 |
|---|---|---|---|---|
| 训练/验证评测 | `TrainValPipeline.run()` | task + dataset + model config | Lightning test metrics/logs/checkpoints | 模型训练与 benchmark |
| Task callbacks/monitor | `BaseTask.get_monitor_cfg()`、callbacks | validation/test predictions | metrics, saved outputs | 具体任务 |
| 工具级打分 | Vina score、QED/SA/LogP/Lipinski/similarity/property prediction | Molecule/Protein/Pocket | score/text | workflow 中即时评价 |

缺口：没有统一“新算法提交 -> 自动标准化输入输出 -> 多数据集评测 -> 指标排行榜 -> 发布到算法仓库”的模块。若要贴合参考图中的“新算法 -> 算法评测器 -> 算法仓库”，需要新增统一算法插件规范和评测流水线。

## 4. 典型模块间箭头实例

### 4.1 结构化药物设计 workflow

以 `configs/workflow/stable_drug_design.yaml` 为例：

```text
输入: protein PDB + reference pocket PKL
  -> visualize_protein_pocket(protein, pocket) -> image
  -> structure_based_drug_design(pocket) -> molecule_0
      -> protein_molecule_docking_score(protein, molecule_0) -> score_0
      -> text_based_molecule_editing(molecule_0, text="lower liver toxicity") -> molecule_1
      -> pocket_molecule_docking(pocket, molecule_1) -> molecule_1_pose
          -> protein_molecule_docking_score(protein, molecule_1_pose) -> score_1
          -> visualize_complex(protein, molecule_1_pose) -> image
      -> molecule_question_answering(molecule_0/molecule_1, property questions) -> text answers
      -> molecule_property_prediction(molecule_0/molecule_1, task=SIDER) -> toxicity/ADR score
```

接口要点：`structure_based_drug_design` 输出 `Molecule`，通过 `wrap_outputs()` 自动以 `molecule` 字段注入下游；`protein`/`pocket` 是节点固定输入或上游输出。

### 4.2 蛋白定向进化 workflow

`configs/workflow/directed_evolution.yaml` 表达：

```text
UniProt accession
  -> protein_uniprot_request(accession) -> Protein
      -> protein_question_answering(Protein, motif/domain question) -> Text
      -> protein_question_answering(Protein, function question) -> Text
      -> mutation_engineering(Protein, desired property text) -> mutation
      -> apply_mutation_to_sequence(Protein, mutation) -> mutated Protein
      -> mutation_engineering(mutated Protein, desired property text) -> mutation
      -> apply_mutation_to_sequence(...)
      -> protein_folding(mutated Protein) -> structured Protein
      -> visualize_protein(structured Protein) -> image
```

其中 mutation 工具输出默认字段是 `output`，workflow 边通过 `name_mapping: output -> mutation` 把它改名成 `apply_mutation_to_sequence` 所需输入字段。

### 4.3 PDB 查询 workflow memory

`memory/workflows/pdb_query.yaml` 表达：

```text
PDB ID
  -> protein_pdb_request(mode=metadata) -> metadata JSON/text
      -> code_execution(JSON dumps + collect) -> content list
      -> summarize_content(content) -> summary
  -> protein_pdb_request(mode=file_only) -> pdb_file
      -> extract_molecules_from_pdb_file(pdb_file) -> proteins/ligands/ions list
```

该 workflow 是 Agent memory，可被 `PlannerExecutor` 在自然语言任务中调用。

## 5. 与参考架构的差距与建议

### 5.1 已具备能力

- 统一 Tool 接口和懒加载工具注册表。
- 统一数据对象，支持对象/文件两种传递形式。
- 预定义 workflow DAG，支持 name mapping 和节点重复执行。
- Agent 能把自然语言、工具说明、workflow memory、代码执行环境连接起来。
- HTTP 服务把外部应用请求转成 Tool 调用。
- 外部数据库/搜索工具覆盖 PubChem、UniProt、PDB、STRING、ChEMBL。

### 5.2 主要缺口

1. **任务队列管理器不足**：当前只有内存 `exec_queue` 和后台 `asyncio.create_task()`，无持久化队列、重试、优先级、任务状态表、用户级隔离。
2. **任务调度器不具备资源调度**：没有 GPU/CPU/容器/节点级排队与资源分配；仅有可选 Docker 和可视化 subprocess。
3. **算法仓库概念分散**：算法散落在 `TASK_REGISTRY`、`MODEL_REGISTRY`、`TOOLS`、`skills`；没有统一元数据、版本、I/O schema、资源需求和评测状态。
4. **算法评测器不统一**：训练评测、工具打分、workflow 评价分散，缺少“新算法接入”标准流程。
5. **应用层不完整**：后端 API 与前端 JSON parser 存在，但仓库没有完整前端应用代码。
6. **合成规划/临床优化主要在 skills 或组合层**：core tools 里没有完整专用执行器。
7. **接口校验有不一致点**：例如 server 的 `import_pocket` 配置声明 required input 是 `pocket, indices`，但 handler 实际读取 `protein, indices`；建议修正为 `protein, indices`。

### 5.3 如果要演进到参考图架构

建议新增或强化：

- `AlgorithmSpec`：统一算法元数据（name、domain、version、inputs、outputs、resource、container、metrics）。
- `AlgorithmRegistry`：把 `TASK_REGISTRY`、`TOOLS`、`skills` 统一到可检索仓库。
- `TaskQueue`：持久化任务表、状态机、重试、优先级、用户配额。
- `Scheduler`：按资源需求把任务发到本机、Docker、K8s、远程 GPU worker。
- `Evaluator`：统一 benchmark 数据集、指标、报告、可视化和回归测试。
- `DataConnector`：将内部数据对象、外部数据库、文件存储、OSS URL 做成一致 schema。
- `WorkflowRuntime`：将当前 YAML DAG 扩展为可恢复、可追踪、可缓存、可审计的 pipeline runtime。

## 6. 快速索引：核心文件

| 文件 | 作用 |
|---|---|
| `open_biomed/core/agent.py` | LLM 任务规划、工具调用、代码执行、报告导出 |
| `open_biomed/core/workflow.py` | 前端图解析、Workflow DAG 构建与执行、边传递 |
| `open_biomed/core/pipeline.py` | 训练/验证 Pipeline、推理 Pipeline、Ensemble Pipeline |
| `open_biomed/tools/tool_registry.py` | 所有可调用工具的懒加载注册表 |
| `open_biomed/tools/base_tool.py` | Tool 抽象接口与 serial execution 装饰器 |
| `open_biomed/tools/web_request_tools.py` | 外部数据库/搜索请求工具 |
| `open_biomed/tools/third_party_tools.py` | 第三方 CLI 工具，例如 P2Rank |
| `open_biomed/tools/tool_misc.py` | pocket 导入、导出、突变应用、摘要、PDB 分子提取等工具 |
| `open_biomed/scripts/inference.py` | 重模型推理工具构造函数 |
| `open_biomed/scripts/run_server.py` | FastAPI 工具服务 |
| `open_biomed/scripts/run_server_workflow.py` | FastAPI workflow/report 服务 |
| `open_biomed/tasks/__init__.py` | 支持的 task registry |
| `open_biomed/data/*` | 运行时数据对象 |
| `configs/workflow/*`、`memory/workflows/*` | 预定义 workflow |
| `skills/*/SKILL.md` | 高层 biomedical skills/算法工作流说明 |
