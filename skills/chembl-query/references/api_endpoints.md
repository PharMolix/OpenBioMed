# ChEMBL API Reference

## API Endpoints

**Base URL:** `https://www.ebi.ac.uk/chembl/api/data`

### Target Endpoints

| Endpoint | Description |
|----------|-------------|
| `/target/search.json?q={query}` | Search targets by name |
| `/target/{chembl_id}.json` | Get target by ChEMBL ID |
| `/target?target_chembl_id__exact={uniprot}.json` | Get target by UniProt ID |

### Molecule Endpoints

| Endpoint | Description |
|----------|-------------|
| `/molecule/search.json?q={name}` | Search molecules by name |
| `/molecule/filter.json?molecule_structures__canonical_smiles__exact={smiles}` | Search by exact SMILES |
| `/molecule/{chembl_id}.json` | Get molecule by ChEMBL ID |

### Activity Endpoints

| Endpoint | Description |
|----------|-------------|
| `/activity.json?target_chembl_id={id}` | Get activities for target |
| `/activity.json?molecule_chembl_id={id}` | Get activities for molecule |

### Drug Indication Endpoints

| Endpoint | Description |
|----------|-------------|
| `/drug_indication.json?mesh_heading__icontains={term}` | Search by disease name |

## Common Filters

### Activity Filters

| Filter | Description | Example |
|--------|-------------|---------|
| `standard_type` | Activity type | `IC50`, `Ki`, `EC50`, `Kd` |
| `standard_value__lte` | Max value (nM) | `1000` |
| `standard_value__gte` | Min value (nM) | `10` |
| `pchembl_value__gte` | Min pChEMBL | `7` |

### Indication Filters

| Filter | Description | Example |
|--------|-------------|---------|
| `mesh_heading__icontains` | Disease name search | `diabetes` |
| `max_phase_for_ind` | Clinical phase | `4` (approved) |

## Response Format

### Activity Response Fields

```json
{
  "molecule_chembl_id": "CHEMBL25",
  "molecule_pref_name": "ASPIRIN",
  "target_chembl_id": "CHEMBL240",
  "target_pref_name": "Cyclooxygenase-2",
  "target_organism": "Homo sapiens",
  "standard_type": "IC50",
  "standard_value": "4260",
  "standard_units": "nM",
  "pchembl_value": "5.37",
  "assay_chembl_id": "CHEMBL644534",
  "assay_description": "Inhibition of COX-2..."
}
```

### Drug Indication Response Fields

```json
{
  "molecule_chembl_id": "CHEMBL1073",
  "molecule_pref_name": "GLIPIZIDE",
  "mesh_heading": "Diabetes Mellitus",
  "max_phase_for_ind": "4.0",
  "efo_term": "diabetes mellitus",
  "drugind_id": "12345"
}
```

## Rate Limits

- Default rate limit: 5 requests per second
- Use `limit` parameter to control result size
- Large result sets are paginated

## Common Issues

### SMILES Search Returns No Results

**Problem:** Exact SMILES match fails

**Solution:**
1. Ensure SMILES is canonical (use RDKit to standardize)
2. Try molecule name search instead
3. Check for stereochemistry differences

### Target Search Returns Wrong Target

**Problem:** Generic name matches multiple targets

**Solution:**
1. Use UniProt ID for precise matching
2. Specify `standard_type` to filter relevant activities
3. Check target organism if needed

### Disease Search Returns No Results

**Problem:** "breast cancer" returns no results

**Solution:**
1. Use simpler terms: "breast" or "neoplasm"
2. ChEMBL uses MeSH headings, not common names
3. Check `efo_term` field for alternative disease names

## Related Resources

- [ChEMBL Web Services Documentation](https://chembl.gitbook.io/chembl-interface-documentation/web-resource-api)
- [ChEMBL API Explorer](https://www.ebi.ac.uk/chembl/api/data/docs)
- [ChEMBL Beaker (Utility Services)](https://chembl.gitbook.io/chembl-interface-documentation/chEMBL-beaker)
