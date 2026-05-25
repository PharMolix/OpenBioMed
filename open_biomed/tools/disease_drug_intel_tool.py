import asyncio
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import aiohttp
from open_biomed.tools.base_tool import Tool


class DiseaseDrugIntelTool(Tool):
    """
    Integrated tool for disease-drug intelligence queries.
    Combines ChEMBL, ClinicalTrials, and Tavily Search APIs.
    """

    # ChEMBL API
    CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
    # ClinicalTrials API
    CLINICALTRIALS_BASE_URL = "https://clinicaltrials.gov/api/v2"

    def __init__(self, timeout: int = 30, max_results: int = 5) -> None:
        super().__init__()
        self.timeout = timeout
        self.max_results = max_results
        self.tavily_api_key = None

    def print_usage(self) -> str:
        return """
Disease-drug intelligence query tool.
Supports query_types:
- chembl_search_target: Search targets in ChEMBL
- chembl_search_molecule: Search molecules/drugs in ChEMBL
- chembl_get_drug: Get drug details by ChEMBL ID
- chembl_get_molecule: Get molecule details by ChEMBL ID
- chembl_get_target: Get target details by ChEMBL ID
- chembl_get_mechanism: Get mechanism of action for a molecule
- chembl_get_indication: Get drug indications
- clinicaltrials_search: Search clinical trials
- clinicaltrials_get: Get a specific clinical trial
- search: Web search via Tavily

Inputs: {"query_type": str, ...additional params based on query_type}
Outputs: {"results": dict}
"""

    async def run_async(
        self, query_type: str, **kwargs
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Async execution for all query types."""
        results = []
        messages = []

        if query_type.startswith("chembl_"):
            result = await self._query_chembl(query_type.replace("chembl_", ""), **kwargs)
            results.append(result)
            messages.append(f"ChEMBL query '{query_type}' completed")
        elif query_type.startswith("clinicaltrials_"):
            result = await self._query_clinicaltrials(query_type.replace("clinicaltrials_", ""), **kwargs)
            results.append(result)
            messages.append(f"ClinicalTrials query '{query_type}' completed")
        elif query_type == "search":
            result = await self._query_tavily(**kwargs)
            results.append(result)
            messages.append("Tavily search completed")
        else:
            raise ValueError(f"Unknown query_type: {query_type}")

        return results, messages

    def run(self, query_type: str, **kwargs) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Sync wrapper for async execution."""
        return asyncio.run(self.run_async(query_type, **kwargs))

    # ==================== ChEMBL Methods ====================

    async def _query_chembl(self, action: str, **kwargs) -> Dict[str, Any]:
        """Query ChEMBL API."""
        endpoint, params = self._build_chembl_request(action, **kwargs)
        return await self._http_get(self.CHEMBL_BASE_URL, endpoint, params)

    def _build_chembl_request(self, action: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Build ChEMBL endpoint and params."""
        limit = kwargs.get("limit", 10)
        offset = kwargs.get("offset", 0)

        if action == "search_target":
            return "target/search", {"q": kwargs.get("query", ""), "limit": limit, "offset": offset}
        elif action == "search_molecule":
            return "molecule/search", {"q": kwargs.get("query", ""), "limit": limit, "offset": offset}
        elif action == "get_drug":
            return f"drug/{kwargs.get('chembl_id', '')}", {}
        elif action == "get_molecule":
            return f"molecule/{kwargs.get('chembl_id', '')}", {}
        elif action == "get_target":
            return f"target/{kwargs.get('chembl_id', '')}", {}
        elif action == "get_mechanism":
            params = {"limit": kwargs.get("limit", 20)}
            if kwargs.get("molecule_chembl_id"):
                params["molecule_chembl_id"] = kwargs["molecule_chembl_id"]
            return "mechanism", params
        elif action == "get_indication":
            params = {"limit": kwargs.get("limit", 20)}
            if kwargs.get("molecule_chembl_id"):
                params["molecule_chembl_id"] = kwargs["molecule_chembl_id"]
            if kwargs.get("efo_term"):
                params["efo_term"] = kwargs["efo_term"]
            return "drug_indication", params
        else:
            raise ValueError(f"Unknown ChEMBL action: {action}")

    # ==================== ClinicalTrials Methods ====================

    async def _query_clinicaltrials(self, action: str, **kwargs) -> Dict[str, Any]:
        """Query ClinicalTrials.gov API."""
        endpoint, params = self._build_clinicaltrials_request(action, **kwargs)
        return await self._http_get(self.CLINICALTRIALS_BASE_URL, endpoint, params)

    def _build_clinicaltrials_request(self, action: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Build ClinicalTrials endpoint and params."""
        if action == "search":
            params = {
                "format": "json",
                "markupFormat": "markdown",
                "pageSize": str(kwargs.get("page_size", 20)),
                "countTotal": str(kwargs.get("count_total", False)).lower(),
            }
            if kwargs.get("query_cond"):
                params["query.cond"] = kwargs["query_cond"]
            if kwargs.get("query_term"):
                params["query.term"] = kwargs["query_term"]
            if kwargs.get("filter_overall_status"):
                params["filter.overallStatus"] = kwargs["filter_overall_status"]
            if kwargs.get("fields"):
                params["fields"] = ",".join(kwargs["fields"])
            if kwargs.get("sort"):
                params["sort"] = ",".join(kwargs["sort"])
            if kwargs.get("page_token"):
                params["pageToken"] = kwargs["page_token"]
            return "studies", params
        elif action == "get":
            nct_id = kwargs.get("nct_id", "")
            params = {"format": "json", "markupFormat": "markdown"}
            if kwargs.get("fields"):
                params["fields"] = ",".join(kwargs["fields"])
            return f"studies/{nct_id}", params
        else:
            raise ValueError(f"Unknown ClinicalTrials action: {action}")

    # ==================== Tavily Search Methods ====================

    async def _query_tavily(self, **kwargs) -> Dict[str, Any]:
        """Query Tavily search API."""
        import os
        try:
            from langchain_tavily import TavilySearch
        except ImportError:
            return {"error": "langchain_tavily is not installed"}

        api_key = kwargs.get("api_key") or os.getenv("TAVILY_API_KEY")
        if not api_key:
            return {"error": "TAVILY_API_KEY is not set"}

        max_results = kwargs.get("max_results", self.max_results)
        query = kwargs.get("query", "")

        tool = TavilySearch(
            tavily_api_key=api_key,
            max_results=max_results,
            topic="general",
            include_answer=True,
        )
        return tool.invoke(query)

    # ==================== HTTP Utilities ====================

    async def _http_get(
        self, base_url: str, endpoint: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform async HTTP GET request."""
        url = f"{base_url}/{endpoint.lstrip('/')}"
        headers = {"Accept": "application/json"}

        # Build query string manually to handle dot notation in parameter names
        if params:
            query_string = urlencode(params, safe='.')
            url = f"{url}?{query_string}"

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=headers, timeout=self.timeout
            ) as response:
                response.raise_for_status()
                return await response.json()