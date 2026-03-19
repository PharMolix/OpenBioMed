---
name: disease-drug-intelligence
description: 面向生物医药问答场景的“疾病到创新药综合分析”技能。用于回答“某疾病有哪些创新药/前沿药/在研药/新机制药”等问题，输出疾病-靶点-药物-临床进展-机制趋势的一体化证据报告。适用于联合本地数据库适配代码与公开数据库 API（ChEMBL、ClinicalTrials、Search）进行多数据库查询、去重归一、创新性筛选与结构化报告生成的任务。
---

# 疾病创新药情报整合

## 概述

将自然语言问题（如“阿尔茨海默病最近有哪些值得关注的新药？”）转换为可执行的多库查询计划。  
生成面向决策的中文综合报告，而非仅返回药名列表。

## 快速开始

1. 识别是否属于 `disease_to_drug` 场景。
2. 标准化疾病实体并拆解“创新药”意图。
3. 优先调用本地 `local_tools/` 代码执行数据库查询。
4. 做实体归一、证据整合、创新性筛选与分层输出。
5. 生成中文报告并标注证据边界。

详细的数据结构、路由表、评分规则、报告模板见 [disease_to_drug_playbook.md](references/disease_to_drug_playbook.md)。

## 本地执行约定

本技能不再依赖 BioDB MCP HTTP 服务。  
涉及 `ChEMBL`、`ClinicalTrials`、`Search` 的调用时，只允许直接使用当前技能目录下的本地 Python 代码：

- `local_tools/chembl_api.py`
- `local_tools/clinicaltrials_api.py`
- `local_tools/search_api.py`
- `local_tools/run_tool.sh`

推荐调用方式：

```bash
bash local_tools/run_tool.sh chembl_api.py search_target EGFR
bash local_tools/run_tool.sh chembl_api.py search_molecule osimertinib
bash local_tools/run_tool.sh chembl_api.py get_drug_by_id CHEMBL3545063
bash local_tools/run_tool.sh clinicaltrials_api.py get_studies --query-cond "lung cancer" --fields NCTId BriefTitle OverallStatus
bash local_tools/run_tool.sh search_api.py "latest EGFR inhibitor approval"
```

执行约束：

- 优先 import 本地 `local_tools/` 模块，或通过 `bash local_tools/run_tool.sh ...` 执行，不再假设 `http://127.0.0.1:8086` 一类 MCP 服务存在。
- 不要直接依赖裸 `python` 命令；统一通过 `run_tool.sh` 解析可用解释器。`run_tool.sh` 会优先使用 `python3`，仅在缺失时才回退到 `python`。
- 若 `Search` 需要运行，必须确认已安装 `langchain_tavily` 且环境变量 `TAVILY_API_KEY` 已设置。
- 当前数据库查询能力仅包含 `ChEMBL`、`ClinicalTrials`、`Search`。与本技能当前实现无关的其他数据库说明应忽略，不参与执行。
- `Search` 的执行只允许通过 `bash local_tools/run_tool.sh search_api.py ...` 或 `SearchAPI.run(query)` 完成；不得绕过本地工具直接调用外部网页搜索。
- 只有在用户明确要求使用外部网页搜索，且同时说明本地 `Search` 工具不可用或结果不足时，才允许把外部网页搜索作为最终兜底；否则一律禁止。

## 触发与判定

当用户问题同时包含以下信息时，触发本技能：
- 疾病实体：如糖尿病、肺癌、阿尔茨海默病、肥胖症、NASH、RA。
- 药物创新意图：如创新药、新机制药、在研药、前沿药、值得关注的新药。

若只提“癌症创新药”等过宽问题，先建议缩小病种；若用户不愿缩小，默认给 Top 癌种与 Top 机制概览。

## 工作流（固定骨架）

### Step 0 问题结构化
构造任务对象（示例）：
```json
{
  "task_type": "disease_to_drug",
  "focus": "innovative_drugs",
  "disease_raw": "糖尿病",
  "time_constraint": null,
  "region_constraint": null,
  "stage_constraint": null
}
```

### Step 1 疾病标准化
输出 `canonical_disease`、`subtypes`、`aliases`、`preferred_query_terms`。  
若用户未指定亚型，先做总疾病分析，再强调研发更活跃亚型（例如 diabetes 下优先覆盖 T2DM）。

### Step 2 创新药定义映射
将“创新药”映射为可执行准则：
- 新机制/新靶点（含 first-in-class 倾向）
- 近年代表性获批药
- 中后期在研候选（II/III 期优先）
- 前沿方向（双/多靶点、新一代优化分子）

### Step 3 子任务拆分
固定执行 5 个子任务：
- `identify_targets_and_mechanisms`
- `retrieve_representative_drugs`
- `build_drug_profiles`
- `validate_clinical_progress`
- `summarize_trends`

### Step 4 数据库执行顺序
默认顺序：
1. 机制与靶点：`ChEMBL(target/mechanism)`
2. 药物候选：`ChEMBL(molecule/drug/indication)`
3. 临床推进：`ClinicalTrials`
4. 网页兜底：`Search`（仅允许使用本地 `search_api.py`，且仅在 `ChEMBL` 与 `ClinicalTrials` 证据不足或需要最新进展核实时）

说明：当前技能不依赖 BioDB MCP 服务；`ChEMBL`、`ClinicalTrials`、`Search` 已切换为本地适配代码。本技能的稳定执行范围仅覆盖这三类能力。

### Step 5 证据整合与去重
优先主键：
- 药物：`ChEMBL ID > 标准药名 > ClinicalTrials intervention`
- 靶点：`基因符号/标准靶点名 > 别名`

必须保留别名和剂型信息，避免错误合并（如 semaglutide 不同制剂）。

### Step 6 创新性筛选与排序
按 0-5 分打分并综合排序：
- `disease_relevance`
- `innovation`
- `clinical_maturity`
- `evidence_strength`
- `representativeness`

输出必须分层：
- 已上市/已验证代表性创新药
- 中后期在研候选药
- 前沿探索机制方向

### Step 7 报告生成
报告生成前，必须先读取 `references/disease_to_drug_playbook.md` 中的 `## 10. 中文报告模板（标准版）`。

默认必须严格按照该模板输出最终中文报告，不得仅做“包含这些内容”的自由发挥。章节顺序、一级编号和主标题骨架必须保持一致：

- `{疾病名称} 创新药综合分析报告`
- `1. 问题概述`
- `2. 结论先行`
- `3. 疾病相关关键靶点与机制`
- `4. 代表性创新药物清单`
- `5. 临床试验进展概览`
- `6. 研发趋势与判断`
- `7. 结果说明与局限`

只有在用户明确要求“简版/摘要版/表格版/特定格式”时，才允许偏离标准模板；若未明确提出，必须使用标准模板。

### Step 8 异常兜底
- 结果过多：按代表性+创新性取 Top N（默认 10）。
- 结果过少：优先输出靶点方向与邻近机制，不强行凑药物列表。
- 证据冲突：明确写出“分子证据存在/临床证据有限”。
- 约束缺失：默认 `time_constraint=null, region_constraint=global`，并在报告中显式声明。

## 工具调用约束

- 先跑主链路（本地 `ChEMBL` + 本地 `ClinicalTrials`），再做补强，不要反过来。
- 若缺失关键字段（phase/status/target），必须触发补查。
- 报告中的每个关键结论至少有一条可追溯证据（数据库名 + 实体主键）。
- 不把“创新药”当监管定义；它是信息整合定义，必须在结果说明中声明。
- 对 `ChEMBL` 的最小必需能力，优先使用 `search_target`、`search_molecule`、`get_drug_by_id`、`get_molecule_by_id`、`get_target_by_id`、`get_mechanism`、`get_drug_indication`。
- 对 `ClinicalTrials` 的最小必需能力，优先使用 `get_studies` 与 `get_study`。
- 对 `Search` 的最小必需能力，统一使用 `SearchAPI.run(query)`；如缺少依赖或 API key，需在结果中声明无法执行该补充检索，不得自动切换到外部网页搜索。
- 命令行执行本地工具时，统一使用 `bash local_tools/run_tool.sh <tool.py> ...`，不要直接写 `python <tool.py> ...`。
- 不要再引入 `KEGG`、`UniProt`、`STRING`、`Ensembl`、`PubChem`、`PDB` 等未纳入当前 skill 代码层的数据库说明或调用要求。
- 输出前必须自检：报告章节是否与 `## 10. 中文报告模板（标准版）` 完全对齐；若不对齐，先重写再输出。

## 质量检查清单

- 是否完成疾病标准化（含别名/亚型）？
- 是否给出机制-药物-临床三层证据链？
- 是否完成实体归一和别名去重？
- 是否分层输出（上市/中后期/前沿）？
- 是否明确局限、冲突与不确定性？

## 参考文件

- [disease_to_drug_playbook.md](references/disease_to_drug_playbook.md)：完整 SOP、路由策略、内部 JSON 结构、中文报告模板。
