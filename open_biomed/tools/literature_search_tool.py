import asyncio
import logging
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from open_biomed.tools.base_tool import Tool


class LiteratureSearchTool(Tool):
    """
    Biomedical literature search tool using PubMed and bioRxiv APIs.
    Searches research papers with titles, abstracts, and metadata.

    NCBI EUtils may be blocked for certain server IPs — in that case,
    the tool returns a structured error message instead of crashing.
    """

    PUBMED_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    BIORXIV_API_BASE = "https://api.biorxiv.org"
    NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")

    def __init__(self, timeout: int = 30, rate_limit_delay: float = 0.33) -> None:
        super().__init__()
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay  # NCBI recommends 3 requests/second

    def print_usage(self) -> str:
        return """
Biomedical literature search tool using PubMed and bioRxiv APIs.
Supports query_types:
- pubmed_search: Search PubMed by keywords
- pubmed_fetch: Fetch paper details by PMID
- biorxiv_fetch: Fetch bioRxiv papers by date range
- biorxiv_category: Fetch bioRxiv papers by category

Inputs: {"query_type": str, "query": str (for pubmed), or "start_date/end_date/days/category" (for biorxiv)}
Outputs: {"results": list of papers with title, abstract, authors, doi, etc.}
"""

    async def run_async(
        self, query_type: str, **kwargs
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Async execution for all query types."""
        results = []
        messages = []

        if query_type == "pubmed_search":
            result = await self._pubmed_search(
                kwargs.get("query", ""),
                max_results=kwargs.get("max_results", 10)
            )
            results.append(result)
            messages.append(f"PubMed search for '{kwargs.get('query', '')}' completed")
        elif query_type == "pubmed_fetch":
            pmids = kwargs.get("pmids", [])
            if isinstance(pmids, str):
                pmids = pmids.split(",")
            result = await self._pubmed_fetch(pmids)
            results.append(result)
            messages.append(f"PubMed fetch for {len(pmids)} PMIDs completed")
        elif query_type == "biorxiv_fetch":
            result = await self._biorxiv_fetch(
                start_date=kwargs.get("start_date"),
                end_date=kwargs.get("end_date"),
                days=kwargs.get("days", 30)
            )
            results.append(result)
            messages.append("bioRxiv fetch by date range completed")
        elif query_type == "biorxiv_category":
            result = await self._biorxiv_fetch(
                start_date=kwargs.get("start_date"),
                end_date=kwargs.get("end_date"),
                days=kwargs.get("days", 30),
                category=kwargs.get("category")
            )
            results.append(result)
            messages.append(f"bioRxiv fetch for category '{kwargs.get('category')}' completed")
        else:
            raise ValueError(f"Unknown query_type: {query_type}")

        return results, messages

    def run(self, query_type: str, **kwargs) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Sync wrapper for async execution."""
        return asyncio.run(self.run_async(query_type, **kwargs))

    # ==================== PubMed Methods ====================

    def _add_ncbi_api_key(self, params: dict) -> dict:
        """Add NCBI API key to request params if available."""
        if self.NCBI_API_KEY:
            params["api_key"] = self.NCBI_API_KEY
        return params

    def _add_ncbi_identifiers(self, params: dict) -> dict:
        """Add `tool` and `email` params so NCBI can identify/contact us.

        Recommended by the E-utilities usage guidelines; lets NCBI reach out
        about our traffic instead of blocking the IP outright.
        """
        params["tool"] = self.EUTILS_TOOL_NAME
        if self.EUTILS_CONTACT_EMAIL:
            params["email"] = self.EUTILS_CONTACT_EMAIL
        return params

    async def _check_ncbi_block(self, response: aiohttp.ClientResponse) -> Optional[str]:
        """Check if NCBI EUtils returned a block/misuse redirect.

        NCBI blocks IPs by returning a 302 redirect to misuse.ncbi.nlm.nih.gov.
        aiohttp follows redirects by default, so we check the final response content.

        Returns an error message if blocked, None if OK.
        """
        # Check URL — if redirected to misuse page
        final_url = str(response.url)
        if "misuse.ncbi.nlm.nih.gov" in final_url:
            return "NCBI EUtils is blocked for this server IP (rate limit or abuse detection). Please try again later or use an alternative data source."

        # Check content — some blocks return HTML directly
        content = await response.text()
        if "misuse.ncbi.nlm.nih.gov" in content or "Abuse" in content and "NCBI" in content:
            return "NCBI EUtils is blocked for this server IP (rate limit or abuse detection). Please try again later or use an alternative data source."

        return None

    async def _pubmed_search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Search PubMed for PMIDs, then fetch paper details."""
        if not query:
            return {"error": "Query is required", "papers": []}

        # Step 1: Search for PMIDs
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json"
        }
        self._add_ncbi_api_key(search_params)
        self._add_ncbi_identifiers(search_params)

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.PUBMED_EUTILS_BASE}/esearch.fcgi",
                params=search_params,
                timeout=self.timeout
            ) as response:
                # Check for NCBI block before processing
                block_msg = await self._check_ncbi_block(response)
                if block_msg:
                    logging.error(f"[LiteratureSearchTool] NCBI EUtils blocked: {block_msg}")
                    return {"error": block_msg, "query": query, "papers": []}
                if response.status != 200:
                    return {"error": f"PubMed ESearch returned HTTP {response.status}", "query": query, "papers": []}
                search_data = await response.json()

        pmids = search_data.get("esearchresult", {}).get("idlist", [])

        if not pmids:
            return {"query": query, "papers_found": 0, "papers": []}

        # Rate limiting
        await asyncio.sleep(self.rate_limit_delay)

        # Step 2: Fetch paper details
        papers = await self._pubmed_fetch(pmids)

        # Propagate block errors from fetch
        if "error" in papers and "blocked" in papers.get("error", "").lower():
            return {"error": papers["error"], "query": query, "papers": []}

        return {
            "query": query,
            "papers_found": len(papers.get("papers", [])),
            "pmids": pmids,
            "papers": papers.get("papers", [])
        }

    async def _pubmed_fetch(self, pmids: List[str]) -> Dict[str, Any]:
        """Fetch paper details from PubMed by PMIDs."""
        if not pmids:
            return {"error": "PMIDs are required", "papers": []}

        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml"
        }
        self._add_ncbi_api_key(fetch_params)
        self._add_ncbi_identifiers(fetch_params)

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.PUBMED_EUTILS_BASE}/efetch.fcgi",
                params=fetch_params,
                timeout=self.timeout
            ) as response:
                # Check for NCBI block before processing
                block_msg = await self._check_ncbi_block(response)
                if block_msg:
                    logging.error(f"[LiteratureSearchTool] NCBI EUtils blocked: {block_msg}")
                    return {"error": block_msg, "pmids": pmids, "papers": []}
                if response.status != 200:
                    return {"error": f"PubMed EFetch returned HTTP {response.status}", "pmids": pmids, "papers": []}
                xml_content = await response.text()

        # Parse XML
        root = ET.fromstring(xml_content)
        papers = []

        for article in root.findall(".//PubmedArticle"):
            paper = self._parse_pubmed_article(article)
            papers.append(paper)

        return {"pmids": pmids, "papers": papers}

    def _parse_pubmed_article(self, article) -> Dict[str, Any]:
        """Parse a PubMed article XML element."""
        # Title
        title_elem = article.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else "N/A"

        # Abstract
        abstract_parts = []
        for abstract_text in article.findall(".//Abstract/AbstractText"):
            text = abstract_text.text or ""
            label = abstract_text.get("Label", "")
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts) if abstract_parts else "No abstract available"

        # Authors
        authors = []
        for author in article.findall(".//Author"):
            lastname = author.find("LastName")
            forename = author.find("ForeName")
            if lastname is not None:
                name = lastname.text
                if forename is not None:
                    name = f"{name} {forename.text}"
                authors.append(name)

        # DOI
        doi = "N/A"
        for article_id in article.findall(".//ArticleId"):
            if article_id.get("IdType") == "doi":
                doi = article_id.text
                break

        # PMID
        pmid_elem = article.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else "N/A"

        # Date
        date = "N/A"
        pub_date = article.find(".//PubDate")
        if pub_date is not None:
            year = pub_date.find("Year")
            month = pub_date.find("Month")
            if year is not None:
                date = year.text
                if month is not None:
                    date = f"{date}-{month.text}"

        # Journal
        journal_elem = article.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else "N/A"

        return {
            "title": title,
            "authors": ", ".join(authors) if authors else "N/A",
            "abstract": abstract,
            "doi": doi,
            "pmid": pmid,
            "date": date,
            "journal": journal,
            "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        }

    # ==================== bioRxiv Methods ====================

    async def _biorxiv_fetch(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30,
        category: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Fetch papers from bioRxiv by date range with retry logic."""
        # Build date range
        if not start_date or not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        # Build URL
        url = f"{self.BIORXIV_API_BASE}/details/biorxiv/{start_date}/{end_date}"

        params = {}
        if category:
            params["category"] = category

        data = {"collection": []}  # Initialize with empty collection

        # Retry logic for rate limiting
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=self.timeout) as response:
                        if response.status == 429:
                            # Rate limited - wait and retry
                            await asyncio.sleep(2 * (attempt + 1))
                            continue
                        response.raise_for_status()
                        data = await response.json()
                        break  # Success, exit retry loop
            except aiohttp.ClientError as e:
                if attempt == max_retries - 1:
                    return {
                        "error": f"bioRxiv API error after {max_retries} retries: {str(e)}",
                        "start_date": start_date,
                        "end_date": end_date,
                        "category": category,
                        "papers_found": 0,
                        "papers": []
                    }
                await asyncio.sleep(1 * (attempt + 1))
                continue

        papers = []
        for item in data.get("collection", []):
            papers.append({
                "title": item.get("title", ""),
                "authors": item.get("authors", ""),
                "abstract": item.get("abstract", ""),
                "doi": item.get("doi", ""),
                "date": item.get("date", ""),
                "category": item.get("category", ""),
                "link": f"https://www.biorxiv.org/content/{item.get('doi', '')}"
            })

        return {
            "start_date": start_date,
            "end_date": end_date,
            "category": category,
            "papers_found": len(papers),
            "papers": papers
        }