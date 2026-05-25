import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import aiohttp
from open_biomed.tools.base_tool import Tool


class DrugDrugInteractionTool(Tool):
    """
    Drug-Drug Interaction (DDI) analysis tool using KEGG DDI database.
    Analyzes potential interactions between up to 5 drugs.
    """

    KEGG_API_BASE = "https://rest.kegg.jp"

    def __init__(self, timeout: int = 30, rate_limit_delay: float = 0.2) -> None:
        super().__init__()
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._drug_cache: Dict[str, Dict[str, Any]] = {}

    def print_usage(self) -> str:
        return """
Drug-Drug Interaction (DDI) analysis tool using KEGG DDI database.
Supports query_types:
- find_drug: Find KEGG drug ID from drug name
- get_drug_info: Get detailed drug information from KEGG
- get_interactions: Query DDI for multiple drugs
- analyze: Complete DDI analysis workflow for up to 5 drugs

Inputs: {"query_type": str, "drugs": List[str] or "query": str, ...}
Outputs: {"results": dict with interactions, drug_info, severity_summary}
"""

    async def run_async(
        self, query_type: str, **kwargs
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Async execution for all query types."""
        results = []
        messages = []

        if query_type == "find_drug":
            result = await self._find_drug(kwargs.get("query", ""))
            results.append(result)
            messages.append(f"Drug ID lookup for '{kwargs.get('query', '')}' completed")
        elif query_type == "get_drug_info":
            result = await self._get_drug_info(kwargs.get("drug_id", ""))
            results.append(result)
            messages.append(f"Drug info for '{kwargs.get('drug_id', '')}' retrieved")
        elif query_type == "get_interactions":
            drug_ids = kwargs.get("drug_ids", [])
            if isinstance(drug_ids, str):
                drug_ids = drug_ids.split(",")
            result = await self._get_interactions(drug_ids)
            results.append(result)
            messages.append(f"DDI query for {len(drug_ids)} drugs completed")
        elif query_type == "analyze":
            drugs = kwargs.get("drugs", [])
            if isinstance(drugs, str):
                drugs = drugs.split(",")
            result = await self._analyze(drugs)
            results.append(result)
            messages.append(f"Complete DDI analysis for {len(drugs)} drugs completed")
        else:
            raise ValueError(f"Unknown query_type: {query_type}")

        return results, messages

    def run(self, query_type: str, **kwargs) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Sync wrapper for async execution."""
        return asyncio.run(self.run_async(query_type, **kwargs))

    # ==================== KEGG API Methods ====================

    async def _kegg_request(self, operation: str, *args) -> str:
        """Make a KEGG API request with rate limiting."""
        url = f"{self.KEGG_API_BASE}/{operation}"
        if args:
            url += "/" + "/".join(str(a) for a in args)

        headers = {"Accept": "text/plain"}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=self.timeout) as response:
                response.raise_for_status()
                text = await response.text()

        # Rate limiting
        await asyncio.sleep(self.rate_limit_delay)
        return text

    async def _find_drug(self, drug_name: str) -> Dict[str, Any]:
        """Find KEGG drug ID from drug name."""
        if not drug_name:
            return {"error": "Drug name is required"}

        result = await self._kegg_request("find", "drug", drug_name)

        if result.strip():
            first_line = result.strip().split('\n')[0]
            parts = first_line.split('\t')
            if len(parts) >= 2:
                kegg_id = parts[0].replace("dr:", "")
                drug_name_found = parts[1] if len(parts) > 1 else ""
                return {
                    "drug_name": drug_name,
                    "kegg_id": kegg_id,
                    "matched_name": drug_name_found.split(";")[0] if drug_name_found else drug_name
                }

        return {"drug_name": drug_name, "kegg_id": None, "error": "Drug not found in KEGG"}

    async def _get_drug_info(self, drug_id: str) -> Dict[str, Any]:
        """Get detailed drug information from KEGG."""
        if not drug_id:
            return {"error": "Drug ID is required"}

        # Check cache
        if drug_id in self._drug_cache:
            return self._drug_cache[drug_id]

        result = await self._kegg_request("get", f"dr:{drug_id}")

        info = {
            "kegg_id": drug_id,
            "name": "",
            "formula": "",
            "targets": [],
            "enzymes": [],
            "atc_codes": []
        }

        for line in result.split('\n'):
            if line.startswith("NAME"):
                info["name"] = line.split(maxsplit=1)[1].strip().rstrip(";")
            elif line.startswith("FORMULA"):
                info["formula"] = line.split(maxsplit=1)[1].strip()
            elif line.startswith("TARGET"):
                info["targets"].append(line.split(maxsplit=1)[1].strip())
            elif line.startswith("METABOLISM"):
                info["enzymes"].append(line.split(maxsplit=1)[1].strip())
            elif line.startswith("REMARK"):
                if "ATC code:" in line:
                    atc_str = line.split("ATC code:")[1].strip()
                    info["atc_codes"].extend(atc_str.split())

        self._drug_cache[drug_id] = info
        return info

    async def _get_interactions(self, drug_ids: List[str]) -> Dict[str, Any]:
        """Query KEGG DDI for drug-drug interactions."""
        if not drug_ids or len(drug_ids) < 2:
            return {"error": "At least 2 drug IDs required", "interactions": []}

        ids = "+".join(drug_ids)
        result = await self._kegg_request("ddi", ids)

        interactions = []
        for line in result.strip().split('\n'):
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                severity_code = parts[2].split(",")[0]
                severity_map = {"CI": "Contraindicated", "P": "Precaution", "C": "Caution"}
                interactions.append({
                    "drug_a_id": parts[0].replace("dr:", ""),
                    "drug_b_id": parts[1].replace("dr:", ""),
                    "severity_code": parts[2],
                    "severity": severity_map.get(severity_code, "Unknown"),
                    "mechanism": parts[3] if len(parts) > 3 else ""
                })

        return {
            "drug_ids": drug_ids,
            "interactions_found": len(interactions),
            "interactions": interactions
        }

    async def _analyze(self, drugs: List[str]) -> Dict[str, Any]:
        """
        Perform complete DDI analysis for multiple drugs.

        Args:
            drugs: List of drug names (up to 5)
        """
        if len(drugs) < 2:
            return {"error": "At least 2 drugs required for DDI analysis"}
        if len(drugs) > 5:
            return {"error": "Maximum 5 drugs allowed per analysis"}

        # Step 1: Resolve drug names to KEGG IDs
        drug_ids = {}
        unresolved = []

        for drug in drugs:
            lookup = await self._find_drug(drug)
            if lookup.get("kegg_id"):
                drug_ids[drug] = lookup["kegg_id"]
            else:
                unresolved.append(drug)

        if not drug_ids:
            return {"error": "No drugs could be resolved", "unresolved": unresolved}

        # Step 2: Get drug information
        drug_info = {}
        for drug, kegg_id in drug_ids.items():
            drug_info[drug] = await self._get_drug_info(kegg_id)

        # Step 3: Query DDI
        ddi_result = await self._get_interactions(list(drug_ids.values()))
        interactions = ddi_result.get("interactions", [])

        # Step 4: Build results
        results = []
        for interaction in interactions:
            drug_a_name = next(
                (d for d, info in drug_info.items() if info.get("kegg_id") == interaction["drug_a_id"]),
                interaction["drug_a_id"]
            )
            drug_b_name = next(
                (d for d, info in drug_info.items() if info.get("kegg_id") == interaction["drug_b_id"]),
                interaction["drug_b_id"]
            )

            results.append({
                "drug_a": f"{drug_a_name} ({interaction['drug_a_id']})",
                "drug_b": f"{drug_b_name} ({interaction['drug_b_id']})",
                "severity": interaction["severity"],
                "severity_code": interaction["severity_code"],
                "mechanism": interaction["mechanism"],
            })

        # Calculate statistics
        severity_counts = {"Contraindicated": 0, "Precaution": 0, "Caution": 0}
        for r in results:
            severity_counts[r["severity"]] = severity_counts.get(r["severity"], 0) + 1

        return {
            "drugs_analyzed": list(drug_ids.keys()),
            "drug_ids": drug_ids,
            "total_pairs": len(drugs) * (len(drugs) - 1) // 2,
            "interactions_found": len(results),
            "severity_summary": severity_counts,
            "interactions": results,
            "drug_details": drug_info,
            "unresolved": unresolved
        }