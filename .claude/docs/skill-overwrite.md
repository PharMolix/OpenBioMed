## Skill Refactoring Guidelines

## 环境
###  改造和验证环境为 docker 容器，具体参见 /scripts/run_docker.sh
###  代码改动后需容器而后进行验证

### Background
Recent commits have been refactoring skills to use the `run_pipeline` and `web_search` APIs.

### Refactoring Rules

1. **Use run_pipeline API**: Skills should call the `/run_pipeline/` endpoint to execute tasks. Use existing tasks and tools whenever possible.

2. **Reuse Before Adding**: Only create new tasks or tools when existing ones cannot meet the skill's requirements. Check `TASK_REGISTRY` and `TOOLS` before implementing new code.

3. **Testing Environment**: Services run in Docker containers. See `scripts/run_docker.sh` for container configuration.

4. **Unit Tests Required**: When adding new tasks, tools, or Python code, add comprehensive unit tests in `test/` directory.

5. **Validation**: After refactoring a skill, simulate user usage to test the skill. The refactored skill must produce results consistent with the original implementation before the refactoring is considered complete.

6. **Error**: 改造过程中遇到的任何报错或者卡点，都需要把完整的上下文以及解决方案总结写入badcase.md ，每次遇到卡点先读badcase.md ，查询是否有类似问题并复用解决方案。

##  重构过程中可以参考已经重构的skills，完成重构的skill需从待重构列表中移到已重构列表
**已重构 Skills (24个)**:
- `target-based-lead-design`, `drug-candidate-discovery`, `pubchem-query`, `admet-prediction`
- `text-based-molecule-editing`, `disease-drug-intelligence`, `kegg-query`, `chembl-query`
- `molecule-biochemical-significance-query-biot5`, `biomedical-literature-search`
- `target-drug-report`, `iupac-name-identification-biot5`, `drug-lead-analysis`
- `uniprot-query`, `ppi-string-query`, `drug-drug-interaction-analysis`
- `retrosynthesis-planning`, `biomed-skill-creator`, `biomed-skill-router`
- `similar-protein-retrieval`, `binding-affinity-prediction-prodigy`
- `mutation-design-aav`, `antibody-structure-prediction-tfold`, `antibody-design-iggm`
- `single-cell-scrna-seq-analysis-scanpy`

**待重构 Skills (21个)**:
1. `cellxgene-census-query`
2. `functional-protein-design`
3. `mutation-design-gfp`
4. `protein-function-prediction`
5. `protein-ligand-binding-analysis-plip`
6. `protein-mutation-analysis`
7. `protein-structure-design-boltzgen`
8. `protein-subcellular-localization-prediction-biot5`
9. `single-cell-atac-seq-peak-calling-annotaion`
10. `single-cell-atac-seq-qc-processing`
11. `single-cell-foundation-model-scrna-seq-geneformer`
12. `single-cell-foundation-model-scrna-seq-langcell`
13. `single-cell-foundation-model-scrna-seq-scgpt`
14. `single-cell-multi-omics-analysis-scvi`
15. `single-cell-multi-omics-data-harmonization`
16. `single-cell-proteomics-data-processing`
17. `single-cell-proteomics-peptide-identification`
18. `spatial-transcriptomics-foundation-model-stofm`
19. `spatial-transcriptomics-spatial-data-io`
20. `structure-prediction-boltz-2`

## 没有明确指令禁止擅自git提交代码
## 每次提交代码必须把更新部分对比完整打印出来，得到批准方可提交