# Disease-to-Innovative-Drug Playbook（中文）

## 1. 任务对象

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

## 2. 疾病标准化输出结构

```json
{
  "canonical_disease": "diabetes mellitus",
  "subtypes": ["type 1 diabetes mellitus", "type 2 diabetes mellitus"],
  "aliases": ["diabetes", "DM", "T1DM", "T2DM"],
  "preferred_query_terms": ["diabetes mellitus", "type 2 diabetes", "type 1 diabetes"]
}
```

## 3. 创新药内部定义

```json
{
  "innovation_definition": {
    "include_new_mechanism": true,
    "include_recent_approved": true,
    "include_late_stage_pipeline": true,
    "include_frontier_candidates": true
  }
}
```

## 4. 固定子任务链

```json
{
  "subtasks": [
    "identify_targets_and_mechanisms",
    "retrieve_representative_drugs",
    "build_drug_profiles",
    "validate_clinical_progress",
    "summarize_trends"
  ]
}
```

## 5. 数据库路由表（按当前 skill 本地工具集）

| 子任务 | 目标 | 主数据库 | 辅数据库 | 输出重点 |
|---|---|---|---|---|
| 疾病标准化 | 标准名称、别名、亚型 | 内部规则 | Search | 标准疾病实体 |
| identify_targets_and_mechanisms | 找关键靶点与机制方向 | ChEMBL | Search | 靶点与机制方向 |
| retrieve_representative_drugs | 拉取代表性药物与候选药 | ChEMBL | 无 | 药物实体、机制、适应症 |
| build_drug_profiles | 构建药物画像 | ChEMBL | 无 | 药物-靶点-机制结构 |
| validate_clinical_progress | 验证临床推进 | ClinicalTrials | Search | phase、status、NCT |
| summarize_trends | 趋势归纳 | ClinicalTrials, ChEMBL | Search | 热门方向与研发格局 |

说明：
- 当前 skill 的稳定执行范围仅覆盖 `ChEMBL`、`ClinicalTrials`、`Search` 三类能力。
- `Search` 仅作为疾病别名补充、最新进展核验和数据库证据不足时的本地工具兜底，不得绕过 skill 自带的 `local_tools/search_api.py` 或 `local_tools/run_tool.sh` 直接调用外部网页搜索。

## 6. 查询优先级与粒度

优先级：
1. 机制/靶点骨架（ChEMBL）
2. 药物候选池（ChEMBL）
3. 临床验证（ClinicalTrials）
4. 本地 Search 补强（Search）

粒度策略：
- 宽泛问题（如“糖尿病创新药”）：先总体，再聚焦高活跃亚型。
- 机制限定（如“GLP-1 创新药”）：先机制锁定，再扩药物。
- 时间限定（如“近五年”）：提高近期临床与近年获批权重。
- 地域限定（如“中国”）：ClinicalTrials + Search 做地域过滤增强。
- 若 `Search` 工具不可用，只能明确说明补充检索未执行，不得自动切换到外部网页搜索。

## 7. 证据整合与去重

药物主键优先级：
1. ChEMBL ID
2. 标准药名
3. ClinicalTrials intervention name（归并后）

靶点主键优先级：
1. gene symbol / 标准 target name
2. 标准 target name
3. 别名

统一药物候选池结构：

```json
[
  {
    "drug_name": "...",
    "aliases": ["..."],
    "target": ["..."],
    "mechanism": "...",
    "evidence": {
      "chembl": {},
      "clinicaltrials": {},
      "search_support": {}
    }
  }
]
```

## 8. 创新性评分（简化版）

```json
{
  "disease_relevance": 0,
  "innovation": 0,
  "clinical_maturity": 0,
  "evidence_strength": 0,
  "representativeness": 0
}
```

输出分层：
- 已上市/已验证代表性创新药
- 中后期在研候选药
- 前沿探索机制方向

## 9. 异常兜底

- 疾病过宽（如“癌症创新药”）：先建议缩小癌种；否则输出 Top 癌种 + Top 机制。
- 库间证据不一致：明确写“机制证据有、临床证据弱/未检出”。
- 结果过多：默认输出 Top N（建议 10）。
- 结果过少：转为“靶点方向 + 邻近机制 + 趋势判断”。
- `Search` 不可用：明确写出 “本次未执行本地 Search 补充检索”，不得自动切换为外部网页搜索。

## 10. 中文报告模板（标准版）

```markdown
{疾病名称} 创新药综合分析报告

1. 问题概述
- 用户关注 {疾病名称} 领域值得关注的创新药物。
- 本报告中“创新药”包含：新机制/新靶点药、近年代表性获批药、中后期在研候选、前沿探索方向。

2. 结论先行
- 当前最值得关注的方向：{方向1}、{方向2}、{方向3}
- 药物分层：
  - 已上市或临床验证较充分的代表性创新药
  - 临床中后期推进中的候选药物
  - 机制前沿但较早期的探索方向

3. 疾病相关关键靶点与机制
3.1 关键靶点
- {靶点1}：{作用简介}
- {靶点2}：{作用简介}
- {靶点3}：{作用简介}

3.2 机制方向总结
- {机制方向1}
- {机制方向2}
- {机制方向3}

4. 代表性创新药物清单
4.1 已上市/较成熟
- 药物：{drug_name}
- 主要靶点/机制：{target/mechanism}
- 适应症关联：{indication}
- 创新点：{why_innovative}
- 临床/上市状态：{status}
- 备注：{remark}

4.2 中后期在研候选
- 药物：{drug_name}
- 主要靶点/机制：{target/mechanism}
- 所处阶段：{phase}
- 值得关注原因：{reason}
- 风险或不确定性：{risk}

4.3 前沿探索方向
- {方向1}：{说明}
- {方向2}：{说明}

5. 临床试验进展概览
5.1 活跃方向
- {方向1}
- {方向2}
- {方向3}

5.2 试验特征
- 常见阶段：{I/II/III}
- 常见干预类型：{small_molecule/peptide/antibody/combination}
- 招募状态概况：{recruiting/active/completed}

6. 研发趋势与判断
- 趋势：{趋势1}、{趋势2}、{趋势3}
- 相对成熟方向：{方向A}
- 前沿探索方向：{方向B}

7. 结果说明与局限
- 本报告是多数据库证据整合结果。
- “创新药”是信息整合口径，不等同于严格监管定义。
- 早研候选药临床证据有限时，需结合本地 Search 工具补充结果与公开来源交叉判断。
```

## 11. 执行硬约束

- 只允许使用当前 skill 提供的本地工具：`local_tools/chembl_api.py`、`local_tools/clinicaltrials_api.py`、`local_tools/search_api.py`。
- 命令行执行时，统一通过 `bash local_tools/run_tool.sh <tool.py> ...` 调用本地工具，不要直接依赖裸 `python` 命令。
- 不要再引入 `KEGG`、`UniProt`、`STRING`、`Ensembl`、`PubChem`、`PDB`、OpenTargets 或任何未纳入当前代码层的数据库要求。
- `Search` 只能通过 `SearchAPI.run(query)` 或 `bash local_tools/run_tool.sh search_api.py ...` 完成。
- 若 `Search` 依赖缺失或 API key 缺失，只能在结果中声明能力不可用，不得自动改用外部网页搜索。
- 输出最终答案前，必须逐项自检是否严格匹配 `## 10. 中文报告模板（标准版）` 的主标题、编号和章节顺序。
