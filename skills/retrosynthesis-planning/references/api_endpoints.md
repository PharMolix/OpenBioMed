# Cheminformatics API Endpoints & Tools Reference

When developing the retrosynthetic pathway, the LLM agent MUST use the following tools to validate concepts and prevent context hallucination:

## Python scripts (`scripts/`)
- **`analyze_molecule.py`**: 
  - `--name <CommonName>` : Resolves chemical name to SMILES using PubChem and outputs Canonical SMILES via RDKit.
  - `--smiles <SMILES>` : Calculates descriptors of the SMILES structure and canonicalizes it.
- **`retro_engine.py`**: 
  - `--retro <SMILES>` : Queries the local AiZynthFinder agent (open source) to fetch high-confidence single-step disconnections.
  - `--vendor <SMILES>` : Hits external supplier APIs to verify if a terminal precursor is commercially available (via PubChem).
- **`route_state_manager.py`**:
  - `--init <Target_Canonical_SMILES>` : Initializes a JSON state file (`route_state.json`) for MCTS/Tree search.
  - `--expand <Parent_Node_ID> --rxn "<Reaction_Name>" --children "<Child_1_SMILES>" ["<Child_2_SMILES>"] --purchasable true false` : Adds expanded branches back into the persistent tree structure to maintain global state.
  - `--prune <Reaction_ID>` : Prunes a specified reaction branch.
  - `--status` : Shows the current status of unsolved leaf nodes.

## Advanced Heuristic Checks (Code Gen)
If you need to validate complex structural constraints (e.g. ring strain, steric clash, protecting group orthogonality) or **forward synthesize** a proposed reverse reaction, you are highly encouraged to **write and run ad-hoc RDKit python code locally**. (e.g. using `rdkit.Chem.AllChem.ReactionFromSmarts`).

## Public APIs for Validation
1. **PubChem REST API** (Compound Data)
   - To get properties by SMILES: `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/<SMILES>/property/MolecularFormula,MolecularWeight,XLogP,IUPACName/JSON`
   - To check availability of compounds: Note that commercial availability is harder to retrieve directly via unauthenticated APIs, so rely on the `analyze_molecule.py` script or prior knowledge.

2. **ChEMBL** API (Optionally used via `curl` for bioactivity or compound lookup):
   - Search by SMILES/Substructure: `https://www.ebi.ac.uk/chembl/api/data/substructure/<SMILES>`
