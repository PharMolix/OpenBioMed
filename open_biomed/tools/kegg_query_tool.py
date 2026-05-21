import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import aiohttp
from ratelimiter import RateLimiter

from open_biomed.tools.base_tool import Tool

KEGG_API_BASE = "https://rest.kegg.jp"


def _auto_format_id(entry_id: str) -> str:
    if any(entry_id.startswith(p) for p in ["dr:", "cpd:", "ds:", "hsa:", "map:", "ko:", "path:", "br:"]:
        return entry_id
    if entry_id.startswith("D"):
        return f"dr:{entry_id}"
    if entry_id.startswith("C"):
        return f"cpd:{entry_id}"
    if entry_id.startswith("H"):
        return f"ds:{entry_id}"
    return entry_id


class KEGGQueryRequester(Tool):
    def __init__(self) -> None:
        super().__init__()
        self.requires_async = True
        self.timeout = 30

    def print_usage(self) -> str:
        return """
Query KEGG database for drug, pathway, and disease information.
Inputs: {"query_type": "find"/"get"/"link", plus type-specific parameters}
Outputs: List of result dicts
"""

    def run(self, query_type: str = "find", **kwargs) -> Tuple[List[Dict], List[str]]:
        return asyncio.run(self.run_async(query_type, **kwargs))

    @RateLimiter(max_calls=5, period=1)
    async def run_async(self, query_type: str = "find", **kwargs) -> Tuple[List[Dict], List[str]]:
        if query_type == "find":
            return await self._find(kwargs.get("database", "drug"), kwargs.get("query", ""), kwargs.get("option"))
        elif query_type == "get":
            entry_id = kwargs.get("entry_id", kwargs.get("query", ""))
            entry_id = _auto_format_id(entry_id)
            return await self._get(entry_id, kwargs.get("option"))
        elif query_type == "link":
            return await self._link(kwargs.get("target_db", "drug"), kwargs.get("source_id", kwargs.get("query", "")))
        else:
            raise ValueError(f"Unknown query_type: {query_type}. Use 'find', 'get', or 'link'.")

    async def _fetch(self, url: str) -> str:
        logging.info(f"[KEGG] Querying: {url}")
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    return content
                else:
                    raise Exception(f"KEGG API returned HTTP {response.status}")

    async def _find(self, database: str, query: str, option: Optional[str] = None) -> Tuple[List[Dict], List[str]]:
        url = f"{KEGG_API_BASE}/find/{database}/{quote(query)}"
        if option:
            url += f"/{option}"
        content = await self._fetch(url)
        results = []
        for line in content.strip().split("\n"):
            if line:
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    results.append({"entry_id": parts[0], "description": parts[1]})
        messages = [json.dumps(results, indent=2)]
        return results, messages

    async def _get(self, entry_id: str, option: Optional[str] = None) -> Tuple[List[Dict], List[str]]:
        url = f"{KEGG_API_BASE}/get/{entry_id}"
        if option:
            url += f"/{option}"
        content = await self._fetch(url)
        result = {"entry_id": entry_id, "raw_text": content}
        messages = [json.dumps(result, indent=2)[:2000]]
        return [result], messages

    async def _link(self, target_db: str, source_id: str) -> Tuple[List[Dict], List[str]]:
        url = f"{KEGG_API_BASE}/link/{target_db}/{source_id}"
        content = await self._fetch(url)
        results = []
        for line in content.strip().split("\n"):
            if line:
                parts = line.split("\t")
                if len(parts) == 2:
                    results.append({"source_id": parts[0], "target_id": parts[1]})
        messages = [json.dumps(results, indent=2)]
        return results, messages
