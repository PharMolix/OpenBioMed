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
**已重构 Skills (22个)**:
- `target-based-lead-design`, `drug-candidate-discovery`, `pubchem-query`, `admet-prediction`
- `text-based-molecule-editing`, `disease-drug-intelligence`, `kegg-query`, `chembl-query`
- `molecule-biochemical-significance-query-biot5`, `biomedical-literature-search`
- `target-drug-report`, `iupac-name-identification-biot5`, `drug-lead-analysis`
- `uniprot-query`, `ppi-string-query`, `drug-drug-interaction-analysis`
- `retrosynthesis-planning`, `biomed-skill-creator`, `biomed-skill-router`
- `similar-protein-retrieval`, `binding-affinity-prediction-prodigy`
- `mutation-design-aav`, `antibody-structure-prediction-tfold`

**待重构 Skills (23个)**:
1. `antibody-design-iggm`
3. `cellxgene-census-query`
4. `functional-protein-design`
5. `mutation-design-gfp`
6. `protein-function-prediction`
7. `protein-ligand-binding-analysis-plip`
8. `protein-mutation-analysis`
9. `protein-structure-design-boltzgen`
10. `protein-subcellular-localization-prediction-biot5`
11. `single-cell-atac-seq-peak-calling-annotaion`
12. `single-cell-atac-seq-qc-processing`
13. `single-cell-foundation-model-scrna-seq-geneformer`
14. `single-cell-foundation-model-scrna-seq-langcell`
15. `single-cell-foundation-model-scrna-seq-scgpt`
16. `single-cell-multi-omics-analysis-scvi`
17. `single-cell-multi-omics-data-harmonization`
18. `single-cell-proteomics-data-processing`
19. `single-cell-proteomics-peptide-identification`
20. `single-cell-scrna-seq-analysis-scanpy`
21. `spatial-transcriptomics-foundation-model-stofm`
22. `spatial-transcriptomics-spatial-data-io`
23. `structure-prediction-boltz-2`

## 没有明确指令禁止擅自git提交代码
## 每次提交代码必须把更新部分对比完整打印出来，得到批准方可提交