# Validation Template

Templates for interactive step-by-step validation during skill creation.

## Step Check-in Template

```
======================================================================
STEP X: [Step Name]
======================================================================
Tool: [tool_name]
Input: [parameters]

Result: [SUCCESS/ERROR]
  - [Key output 1]
  - [Key output 2]
  - [Any errors or warnings]

----------------------------------------------------------------------
SUMMARY: [1-2 sentence summary of what happened]

Is this step result satisfactory? (yes/proceed/modify/skip)
======================================================================
```

## Example: Successful Step

```
======================================================================
STEP 1: RETRIEVING PROTEIN FROM UNIPROT
======================================================================
Tool: protein_uniprot_request
Input: accession='P04637'

Result: SUCCESS
  - Protein object created
  - Name: TP53 (Tumor protein p53)
  - Sequence length: 393 amino acids

----------------------------------------------------------------------
SUMMARY: Successfully retrieved TP53 protein from UniProt database.

Is this step result satisfactory? (yes/proceed/modify)
======================================================================
```

## Example: Failed Step

```
======================================================================
STEP 2: EXPLAINING MUTATION
======================================================================
Tool: mutation_explanation (MutaPLM)
Input: mutation='R248Q'

Result: ERROR
  - Model checkpoint not found
  - Path: ./checkpoints/server/mutaplm.pth

----------------------------------------------------------------------
SUMMARY: ML model unavailable. Need fallback or skip this step.

Is this step result satisfactory? (yes/proceed/modify/skip)
======================================================================
```

## Error Handling Options

When a step fails, present options:

```
Step X failed due to [reason]. Options:

1. **Proceed with fallback** - Use alternative tool/approach
2. **Skip this step** - Continue without this analysis
3. **Modify workflow** - Change the approach entirely
4. **Retry** - Try again (if it might be a transient error)
```

## Final Summary Template

After all steps complete:

```
The validation is complete. Summary of results:
- Step 1: ✅ SUCCESS
- Step 2: ⚠️ FALLBACK USED
- Step 3: ❌ ERROR (skipped)
- Step 4: ✅ SUCCESS

Do you want to:
1. **Proceed** with this workflow (including fallbacks)?
2. **Modify** the workflow and re-validate?
3. **Try a different input** to test other scenarios?
```

## Status Indicators

| Icon | Status | Meaning |
|------|--------|---------|
| ✅ | SUCCESS | Step completed as expected |
| ⚠️ | FALLBACK | Used alternative approach |
| ❌ | ERROR | Step failed, skipped |
| ⏳ | PENDING | Awaiting user input |
