# Skill Refactoring Progress

## Design Principle

/run_pipeline/ calls tasks, handlers use tools. Each skill refactoring creates a new task that calls a corresponding tool.

## Completed Skills

| Skill | Task Name | Status | Verification |
|-------|-----------|--------|--------------|
| binding-affinity-prediction-prodigy | binding_affinity | Done | API tested, PRODIGY tool functional |
| similar-protein-search | similar_protein_search | Done | API tested, FAISS search functional |

## In Progress

| Skill | Task Name | Status | Notes |
|-------|-----------|--------|-------|
| antibody-structure-prediction-tfold | antibody_structure | Implementation Complete | Waiting for tFold model download (ESM-PPI 650M ~2.43GB) |
| antibody-design-iggm | antibody_design | Implementation Complete | Needs IgGM installation for functional verification |

## Pending Skills

Remaining skills to refactor from the original 45 skills in `skills/` directory.

## Files Modified Per Skill

Each skill refactoring typically requires:
1. `open_biomed/tools/third_party_tools.py` - Add Tool class
2. `open_biomed/tools/tool_registry.py` - Register tool
3. `open_biomed/scripts/run_server.py` - Add TaskRequest fields, handler, TASK_CONFIGS
4. `test/api_test.py` - Add unit test
5. `skills/<skill-name>/SKILL.md` - Refactor to use /run_pipeline/ API

## Current Work Details

### antibody-structure-prediction-tfold

**Task**: antibody_structure

**Files Modified**:
- `open_biomed/tools/third_party_tools.py`: Added TFoldAntibodyStructure tool class
- `open_biomed/tools/tool_registry.py`: Added "antibody_structure" registration
- `open_biomed/scripts/run_server.py`: Added heavy_chain, light_chain, antigen fields and handle_antibody_structure handler
- `test/api_test.py`: Added antibody_structure test case
- `skills/antibody-structure-prediction-tfold/SKILL.md`: Refactored with curl examples

**Verification Status**:
- API routing verified (returns proper error/response)
- tFold model download in progress (ESM-PPI 650M ~2.43GB)
- Functional verification pending after model download completes

**Next Steps**:
1. Wait for model download to complete
2. Test API with antibody sequences
3. Commit code to skill branch after full verification