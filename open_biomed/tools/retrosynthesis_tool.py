import json
import logging
import asyncio
from typing import Dict, List, Tuple
from urllib.parse import quote

import aiohttp

from open_biomed.tools.base_tool import Tool


PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


class RetrosynthesisRequester(Tool):
    def __init__(self) -> None:
        super().__init__()
        self.requires_async = True
        self.timeout = 30

    def print_usage(self) -> str:
        return """
Retrosynthesis planning tool.
Inputs: {"query_type": "analyze"/"retro"/"vendor", plus type-specific parameters}
Outputs: List of result dicts
"""

    def run(self, query_type: str = "analyze", **kwargs) -> Tuple[List[Dict], List[str]]:
        return asyncio.run(self.run_async(query_type, **kwargs))

    async def run_async(self, query_type: str = "analyze", **kwargs) -> Tuple[List[Dict], List[str]]:
        if query_type == "analyze":
            return await self._analyze(kwargs.get("query", ""), kwargs.get("molecule", ""))
        elif query_type == "retro":
            return await self._retro(kwargs.get("query", kwargs.get("molecule", "")))
        elif query_type == "vendor":
            return await self._vendor(kwargs.get("query", kwargs.get("molecule", "")))
        else:
            raise ValueError(f"Unknown query_type: {query_type}. Use 'analyze', 'retro', or 'vendor'.")

    async def _fetch(self, url: str) -> str:
        logging.info(f"[Retrosynthesis] Querying: {url}")
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    raise Exception(f"API returned HTTP {response.status}")

    async def _analyze(self, name: str, smiles: str) -> Tuple[List[Dict], List[str]]:
        """Normalize molecule: resolve name to SMILES via PubChem, compute RDKit properties."""
        source = "Direct SMILES"
        target_smiles = smiles

        if name and not target_smiles:
            url = f"{PUBCHEM_BASE}/compound/name/{quote(name)}/property/CanonicalSMILES/JSON"
            content = await self._fetch(url)
            data = json.loads(content)
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                target_smiles = props[0].get("CanonicalSMILES")
                source = f"PubChem Resolution ('{name}')"
            else:
                return [{"error": f"Could not resolve name '{name}' via PubChem."}], [json.dumps({"error": f"Name '{name}' not found"})]

        if not target_smiles:
            return [{"error": "Provide either 'query' (name) or 'molecule' (SMILES)."}], [json.dumps({"error": "No input"})]

        result = self._compute_properties(target_smiles)
        if "error" not in result:
            result["source"] = source
        return [result], [json.dumps(result, indent=2)]

    def _compute_properties(self, smiles: str) -> Dict:
        """Compute RDKit properties for a SMILES string."""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski
        except ImportError:
            return {"error": "RDKit is not installed on the server."}

        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {"error": f"Invalid SMILES: {smiles}"}

        canonical = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hba = Lipinski.NumHAcceptors(mol)
        hbd = Lipinski.NumHDonors(mol)
        lipinski_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])

        return {
            "input_smiles": smiles,
            "canonical_smiles": canonical,
            "physicochemical": {
                "mol_wt": round(mw, 2),
                "exact_mass": round(Descriptors.ExactMolWt(mol), 4),
                "logp": round(logp, 2),
                "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),
            },
            "structural": {
                "num_atoms": mol.GetNumAtoms(),
                "num_heavy_atoms": mol.GetNumHeavyAtoms(),
                "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
                "stereocenters": len(chiral_centers),
            },
            "drug_likeness": {
                "hba": hba,
                "hbd": hbd,
                "lipinski_violations": lipinski_violations,
                "lipinski_pass": lipinski_violations <= 1,
            },
        }

    async def _retro(self, smiles: str) -> Tuple[List[Dict], List[str]]:
        """Run AiZynthFinder for retrosynthetic analysis."""
        try:
            from aizynthfinder.aizynthfinder import AiZynthFinder
            import os
            import platform

            data_dir = os.environ.get("AIZYNTH_DATA_DIR",
                                       "/tmp" if platform.system() != "Windows" else "C:/tmp")
            model_path = os.path.join(data_dir, "uspto_model.onnx")
            templates_path = os.path.join(data_dir, "uspto_templates.csv.gz")

            if not os.path.exists(model_path) or not os.path.exists(templates_path):
                error = {"error": f"Model files missing in {data_dir}. Run `download_public_data {data_dir}` first."}
                return [error], [json.dumps(error)]

            config_dict = {
                "expansion": {"uspto": [model_path, templates_path]},
                "search": {"time_limit": 10, "iteration_limit": 10},
            }
            finder = AiZynthFinder(configdict=config_dict)
            finder.expansion_policy.select("uspto")
            finder.target_smiles = smiles
            finder.tree_search(show_progress=False)
            finder.build_routes()
            routes = getattr(finder.routes, "dicts", getattr(finder.routes, "dictionaries", []))

            results = []
            for route in routes[:10]:
                if "children" in route:
                    for child in route["children"]:
                        precursors = [c["smiles"] for c in child.get("children", [])]
                        if precursors:
                            results.append({
                                "reaction": child.get("metadata", {}).get("type", "Unknown"),
                                "precursors": precursors,
                                "score": route.get("score", 0.5),
                            })

            if not results:
                return [{"status": "no_routes", "message": "No retrosynthetic routes found."}], \
                       [json.dumps({"status": "no_routes"})]
            return results, [json.dumps(results, indent=2)]

        except ImportError:
            error = {"error": "aizynthfinder is not installed on the server."}
            return [error], [json.dumps(error)]
        except Exception as e:
            error = {"error": f"AiZynthFinder failed: {str(e)}"}
            return [error], [json.dumps(error)]

    async def _vendor(self, smiles: str) -> Tuple[List[Dict], List[str]]:
        """Check commercial availability via PubChem CID lookup."""
        url = f"{PUBCHEM_BASE}/compound/smiles/{quote(smiles)}/cids/JSON"
        content = await self._fetch(url)
        data = json.loads(content)
        cids = data.get("IdentifierList", {}).get("CID", [])
        is_purchasable = bool(cids)

        result = {
            "query": smiles,
            "is_purchasable_proxy": is_purchasable,
            "pubchem_cids": cids[:5] if cids else [],
        }
        return [result], [json.dumps(result, indent=2)]