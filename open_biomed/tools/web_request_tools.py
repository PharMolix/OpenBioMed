from abc import abstractmethod, ABC
from typing import Any, Dict, List, Optional, Tuple
import os
import sys
import requests
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import aiohttp
import asyncio
from datetime import datetime
import json
import logging
import random
import re
import threading
import tarfile
from ratelimiter import RateLimiter
from urllib.parse import quote
import xml.etree.ElementTree as ET

from open_biomed.data import Molecule, Protein
from open_biomed.tools.base_tool import Tool

# ==================== Global PubChem Rate Limiter ====================

class PubChemRateLimiter:
    """Global, cross-instance rate limiter for all PubChem API requests.

    PubChem PUG REST limits:
    - Without API key: 2 requests/second
    - With API key: 5 requests/second

    This limiter enforces a global cap regardless of how many requester
    instances exist, and tracks consecutive block events to enforce
    exponential backoff cooldowns when PubChem returns HTML error pages.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        # Global request rate: max 3 per second (conservative for shared server)
        self._min_interval = 0.34  # seconds between requests (~3/sec)
        self._last_request_time = 0.0
        self._request_lock = asyncio.Lock() if hasattr(asyncio, 'Lock') else None
        # Block tracking: consecutive HTML block responses
        self._consecutive_blocks = 0
        self._block_cooldown_until = 0.0  # timestamp: no requests until this time
        # Cooldown progression: 30s, 60s, 120s, 240s, 480s, 600s
        self._cooldown_schedule = [30, 60, 120, 240, 480, 600]

    async def acquire(self):
        """Wait until it's safe to make a PubChem request.

        Enforces:
        1. Global rate limit (min interval between requests)
        2. Block cooldown (exponential backoff after consecutive blocks)
        """
        # Check if we're in a cooldown period from being blocked
        now = time.time()
        if self._block_cooldown_until > now:
            wait = self._block_cooldown_until - now
            logging.warning(f"[PubChemRateLimiter] In cooldown period, waiting {wait:.1f}s before next request")
            await asyncio.sleep(wait)

        # Enforce minimum interval between requests
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            wait = self._min_interval - elapsed
            await asyncio.sleep(wait)

        self._last_request_time = time.time()

    def record_success(self):
        """Record a successful PubChem response — resets block counter."""
        self._consecutive_blocks = 0
        self._block_cooldown_until = 0.0

    def record_block(self):
        """Record a blocked/rate-limited PubChem response.

        Increases consecutive block count and sets cooldown period
        using exponential backoff.
        """
        self._consecutive_blocks += 1
        idx = min(self._consecutive_blocks - 1, len(self._cooldown_schedule) - 1)
        cooldown = self._cooldown_schedule[idx]
        self._block_cooldown_until = time.time() + cooldown
        logging.warning(
            f"[PubChemRateLimiter] Blocked by PubChem (consecutive={self._consecutive_blocks}), "
            f"entering {cooldown}s cooldown. All PubChem requests will be delayed."
        )

    @property
    def is_in_cooldown(self) -> bool:
        return self._block_cooldown_until > time.time()

    @property
    def consecutive_blocks(self) -> int:
        return self._consecutive_blocks


# Module-level singleton
_pubchem_limiter = PubChemRateLimiter()

# ==================== NCBI Error Page Parser ====================

def parse_ncbi_error_html(html_content: str) -> str:
    """Parse NCBI HTML error pages to extract the actual diagnostic message.

    NCBI returns pages like:
    <title>NCBI - WWW Error Blocked Diagnostic</title>
    ... <div class="error-msg">...</div> ...

    Returns a human-readable error summary.
    """
    # Try to extract the title
    title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else "NCBI Error"

    # Try to extract error body text (between divs, paragraphs, etc.)
    # Remove all HTML tags and get text content
    text = re.sub(r'<[^>]+>', ' ', html_content)
    text = re.sub(r'\s+', ' ', text).strip()

    # Truncate to a useful length
    if len(text) > 300:
        text = text[:300] + "..."

    return f"{title}: {text}"


# ==================== IQS Client (Module-level singleton) ====================

class IQSClientWrapper:
    """Alibaba Cloud IQS client wrapper (singleton pattern)"""
    def __init__(self):
        self.client = None
        self.iqs_models = None
        self._try_init()

    def _try_init(self):
        """Initialize IQS client"""
        try:
            from alibabacloud_iqs20241111 import models as iqs_models
            from alibabacloud_iqs20241111.client import Client
            from alibabacloud_tea_openapi import models as open_api_models

            self.iqs_models = iqs_models
            self.open_api_models = open_api_models

            ak = os.environ.get("ALIYUN_AK")
            sk = os.environ.get("ALIYUN_SK")
            endpoint = os.environ.get("ALIYUN_ENDPOINT", "iqs.cn-zhangjiakou.aliyuncs.com")

            if ak and sk:
                config = open_api_models.Config(
                    access_key_id=ak,
                    access_key_secret=sk
                )
                config.endpoint = endpoint
                config.read_timeout = 30000    # 30 seconds
                config.connect_timeout = 10000  # 10 seconds
                self.client = Client(config)
            else:
                logging.warning("ALIYUN_AK or ALIYUN_SK not set, IQS client disabled")
        except ImportError:
            logging.warning("alibabacloud_iqs20241111 not installed, IQS client disabled")

    async def search(self, query: str, search_size: int = 10):
        """Call IQS search API"""
        from Tea.exceptions import TeaException

        if not self.client:
            return []

        request = self.iqs_models.UnifiedSearchRequest(
            body=self.iqs_models.UnifiedSearchInput(
                query=query,
                time_range="NoLimit",
                contents=self.iqs_models.RequestContents(
                    summary=False,
                    main_text=True,
                )
            )
        )

        try:
            response = await self.client.unified_search_async(request)
            results = []
            for item in response.body.page_items:
                results.append({
                    "title": item.title,
                    "text": item.main_text,
                    "url": item.link,
                    "channel": "WebSearch"
                })
            return results[:search_size]
        except TeaException as e:
            logging.error(f"IQS search error: {e}")
            return []

# Module-level singleton
_iqs_client = IQSClientWrapper()

class Requester(Tool):
    def __init__(self) -> None:
        self.requires_async = True

class DBRequester(Requester):
    def __init__(self, timeout: int=30) -> None:
        super().__init__()
        self.timeout = timeout

    def print_usage(self) -> str:
        return "\n".join([
            'Usage: Query a database.',
            'Inputs: {"accession": str (the accession of the database)}',
            "Outputs: Any (the parsed output of the database)"
        ])

    def run(self, accession: Any, **kwargs) -> Any:
        return asyncio.run(self.run_async(accession, **kwargs))

    def _is_pubchem_requester(self) -> bool:
        """Check if this requester hits PubChem APIs (for rate limiting)."""
        return isinstance(self, PubChemRequester) or isinstance(self, PubChemBioactivityRequester)

    async def run_async(self, accession: Any, **kwargs) -> Any:
        url = self._determine_query_url(accession, **kwargs)
        logging.info(f"[DBRequester] Querying: {url} (timeout={self.timeout}s)")

        # Use PubChem global rate limiter for PubChem requests
        max_retries = 3 if self._is_pubchem_requester() else 1

        for attempt in range(max_retries):
            # Acquire PubChem rate limiter if applicable
            if self._is_pubchem_requester():
                await _pubchem_limiter.acquire()

            start_time = time.time()
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            content = await response.read()
                            content = content.decode("utf-8")
                            if content.strip().startswith("<"):
                                error_msg = parse_ncbi_error_html(content)
                                elapsed = time.time() - start_time
                                logging.warning(f"[DBRequester] Received HTML error page from {url}: {error_msg}")
                                if self._is_pubchem_requester():
                                    _pubchem_limiter.record_block()
                                    if attempt < max_retries - 1:
                                        # The cooldown is handled by the rate limiter's acquire()
                                        logging.info(f"[DBRequester] Retry attempt {attempt + 1}/{max_retries} after cooldown")
                                        continue
                                raise Exception(error_msg)
                            # Success! Reset PubChem block counter if applicable
                            if self._is_pubchem_requester():
                                _pubchem_limiter.record_success()
                            elapsed = time.time() - start_time
                            logging.info(f"[DBRequester] Downloaded results successfully in {elapsed:.2f}s")
                        else:
                            logging.warning(f"HTTP request failed, status {response.status}")
                            raise Exception(f"HTTP {response.status}")
            except asyncio.TimeoutError as e:
                elapsed = time.time() - start_time
                logging.error(f"[DBRequester] TIMEOUT: {url} after {elapsed:.2f}s (timeout setting: {self.timeout}s)")
                raise asyncio.TimeoutError(f"Request to {url} timed out after {elapsed:.2f}s")
            except Exception as e:
                elapsed = time.time() - start_time
                # Don't re-log if we're about to retry
                if attempt >= max_retries - 1:
                    logging.error(f"[DBRequester] FAILED: {url} in {elapsed:.2f}s - {e}")
                raise e

        return self._parse_and_save_outputs(accession, content, **kwargs)

    def _parse_and_save_outputs(self, accession: str="", content: str="", **kwargs) -> Any:
        # Parse the content and save them at a local file
        return [content], [content]

    def _determine_query_url(self, accession: str="", **kwargs) -> str:
        if hasattr(self, "db_url"):
            return self.db_url.format(accession=accession)
        else:
            raise NotImplementedError

class MetaDataParser(ABC):
    @abstractmethod
    def parse(self, content: str="", **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def output_format(self) -> str:
        raise NotImplementedError

class RawParser(MetaDataParser):
    def __init__(self, output_format: str="") -> None:
        self.format = output_format

    def parse(self, content: str="", **kwargs) -> Any:
        return content

    def output_format(self) -> str:
        return self.format

# TODO: add metadata parser for PubChem
class PubChemRequester(DBRequester):
    """PubChem molecule query — uses global PubChemRateLimiter for rate control."""

    def __init__(self,
        timeout: int=30
    ) -> None:
        super().__init__(timeout)

    def print_usage(self) -> str:
        return "\n".join([
            'PubChem query.',
            'Inputs: {"accession": a PubChem ID or molecule name}',
            "Outputs: A molecule from PubChem."
        ])

    def _determine_query_url(self, accession: str="", **kwargs) -> str:
        try:
            id = int(accession)
            db_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{accession}/SDF"
        except ValueError:
            db_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{accession}/SDF"
        url = db_url.format(accession=accession)
        api_key = os.environ.get("PUBCHEM_API_KEY")
        if api_key:
            url += f"?api_key={api_key}"
        return url

    def _parse_and_save_outputs(self, accession: str="", content: str="", **kwargs) -> Tuple[List[Molecule], List[str]]:
        if content.strip().startswith("<") or content.strip().startswith("{"):
            raise ValueError(f"PubChem returned non-SDF response for '{accession}'. Content may be an error page or JSON.")
        sdf_file = f"./tmp/pubchem_{accession}.sdf"
        with open(sdf_file, "w") as f:
            f.write(content)
        molecule = Molecule.from_sdf_file(sdf_file)
        return [molecule], [molecule.save_binary()]

class PubChemStructureRequester(Requester):
    """PubChem similarity search — uses global PubChemRateLimiter for rate control."""

    def __init__(self,
        timeout: int=30
    ) -> None:
        super().__init__()
        self.db_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/{accession}/cids/JSON?Threshold={threshold}&MaxRecords={max_records}"
        self.molecule_requester = PubChemRequester()
        self.timeout = timeout

    def print_usage(self) -> str:
        return "\n".join([
            'Usage: Query PubChem for similar molecules.',
            'Inputs: {"accession": str (could be a PubChem ID, SMILES string, or molecular name)}',
            'Outputs: Molecule (an OpenBioMed Molecule object)'
        ])

    def run(self, molecule: Molecule=None, threshold: float=0.8, max_records=10) -> Tuple[List[Molecule], List[str]]:
        return asyncio.run(self.run_async(molecule, threshold, max_records))

    async def run_async(self, molecule: Molecule=None, threshold: float=0.8, max_records=10) -> Tuple[List[Molecule], List[str]]:
        molecule._add_smiles()

        # Check if PubChem is in cooldown — raise error instead of fake data
        if _pubchem_limiter.is_in_cooldown:
            raise Exception("PubChem is in cooldown (blocked/rate-limited). Please wait before retrying.")

        # Acquire global rate limiter
        await _pubchem_limiter.acquire()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    url = self.db_url.format(accession=molecule.smiles, threshold=int(threshold * 100), max_records=max_records)
                    api_key = os.environ.get("PUBCHEM_API_KEY")
                    if api_key:
                        url += f"&api_key={api_key}"
                    async with session.get(url.replace("#", "%23")) as response:
                        if response.status == 200:
                            raw = await response.read()
                            text = raw.decode("utf-8").strip()
                            if not text:
                                logging.warning("PubChem returned empty response for similarity search")
                                _pubchem_limiter.record_success()
                                return [molecule], [molecule.save_binary()]
                            # Check for NCBI HTML error page
                            if text.startswith("<"):
                                error_msg = parse_ncbi_error_html(text)
                                logging.warning(f"[PubChemStructureRequester] Received HTML error page: {error_msg}")
                                _pubchem_limiter.record_block()
                                if attempt < max_retries - 1:
                                    logging.info(f"[PubChemStructureRequester] Retry attempt {attempt + 1}/{max_retries} after cooldown")
                                    await _pubchem_limiter.acquire()
                                    continue
                                # All retries exhausted — raise error
                                raise Exception("PubChem API is blocked or rate-limited after 3 retries. Please try again later.")
                            try:
                                content = json.loads(text)
                            except json.JSONDecodeError:
                                logging.warning(f"PubChem returned non-JSON response: {text[:200]}")
                                _pubchem_limiter.record_success()
                                return [molecule], [molecule.save_binary()]
                            _pubchem_limiter.record_success()
                            logging.info("Downloaded results successfully")
                        elif response.status == 404:
                            _pubchem_limiter.record_success()
                            logging.info("No similar structures found!")
                            return [molecule], [molecule.save_binary()]
                        else:
                            logging.warning(f"HTTP request failed, status {response.status}")
                            raise Exception(f"HTTP {response.status}")
            except Exception as e:
                content = None
                logging.error(f"Download failed. Exception: {e}")
                raise e
            break  # Success — exit retry loop

        all_mols, all_files = [], []
        for cid in content['IdentifierList']['CID']:
            mol, mol_file = await self.molecule_requester.run_async(cid)
            all_mols.extend(mol)
            all_files.extend(mol_file)
        return all_mols, all_files

class PubChemBioactivityRequester(Requester):
    """PubChem bioactivity query — uses global PubChemRateLimiter for rate control.

    Supports:
    1. Query by target gene symbol -> get assays -> get active compounds
    2. Query by compound CID -> get assays where it was tested
    3. Get bioactivity data for a specific assay
    """
    def __init__(self, timeout: int=30) -> None:
        super().__init__()
        self.timeout = timeout
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    def print_usage(self) -> str:
        return "\n".join([
            'PubChem bioactivity query.',
            'Inputs:',
            '  {"query_type": "target", "gene_symbol": "HMGCR"} - Get assays targeting a gene',
            '  {"query_type": "compound", "cid": 2244, "aids_type": "active"} - Get assays for a compound',
            '  {"query_type": "assay", "aid": 1053202, "cids_type": "active"} - Get compounds from an assay',
            '  {"query_type": "bioactivity", "aid": 1053202} - Get full bioactivity data for an assay',
            "Outputs: Dict with bioactivity information"
        ])

    def run(self, query_type: str = "compound", **kwargs) -> Tuple[List[Dict], List[str]]:
        return asyncio.run(self.run_async(query_type, **kwargs))

    def _append_api_key(self, url: str) -> str:
        """Append PubChem api_key to URL if available."""
        api_key = os.environ.get("PUBCHEM_API_KEY")
        if api_key:
            separator = "&" if "?" in url else "?"
            url += f"{separator}api_key={api_key}"
        return url

    async def _pubchem_get(self, session, url: str) -> Tuple[bool, bytes]:
        """Make a PubChem GET request with rate limiting and HTML error detection.

        Returns (is_error_page, raw_content).
        If is_error_page is True, raw_content contains the HTML error text.
        Raises an exception with parsed error message if blocked.
        """
        url = self._append_api_key(url)
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                text = content.decode("utf-8")
                # Detect NCBI HTML error pages (rate-limit/block diagnostic)
                if text.strip().startswith("<"):
                    error_msg = parse_ncbi_error_html(text)
                    _pubchem_limiter.record_block()
                    raise Exception(error_msg)
                _pubchem_limiter.record_success()
                return False, content
            elif response.status in (503, 429):
                # Explicit rate limit response
                _pubchem_limiter.record_block()
                raise Exception(f"PubChem rate limited (HTTP {response.status}). Please retry later.")
            else:
                raise Exception(f"PubChem request failed with HTTP {response.status}")

    async def run_async(self, query_type: str = "compound", **kwargs) -> Tuple[List[Dict], List[str]]:
        # Acquire global rate limiter
        await _pubchem_limiter.acquire()

        results = []
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    if query_type == "target":
                        results = await self._query_by_target(session, **kwargs)
                    elif query_type == "compound":
                        results = await self._query_by_compound(session, **kwargs)
                    elif query_type == "assay":
                        results = await self._query_by_assay(session, **kwargs)
                    elif query_type == "bioactivity":
                        results = await self._query_by_bioactivity(session, **kwargs)
                    else:
                        raise ValueError(f"Unknown query_type: {query_type}. Use 'target', 'compound', 'assay', or 'bioactivity'")
                break  # Success — exit retry loop
            except Exception as e:
                error_str = str(e)
                if "Blocked" in error_str or "rate limited" in error_str:
                    if attempt < max_retries - 1:
                        logging.warning(f"[PubChemBioactivityRequester] Blocked, retry attempt {attempt + 1}/{max_retries}")
                        await _pubchem_limiter.acquire()
                        continue
                    # All retries exhausted — raise with clear message
                    raise Exception(
                        f"PubChem API is blocked or rate-limited after {max_retries} retries. "
                        f"Last error: {error_str}. Please wait a few minutes and try again."
                    )
                raise  # Non-rate-limit error, don't retry

        return results, [json.dumps(results, indent=2)]

    async def _query_by_target(self, session, **kwargs) -> List[Dict]:
        """Get assays targeting a gene."""
        gene_symbol = kwargs.get("gene_symbol")
        gene_id = kwargs.get("gene_id")

        if gene_symbol:
            url = f"{self.base_url}/assay/target/genesymbol/{gene_symbol}/aids/JSON"
        elif gene_id:
            url = f"{self.base_url}/assay/target/geneid/{gene_id}/aids/JSON"
        else:
            raise ValueError("Either gene_symbol or gene_id must be provided for target query")

        _, content = await self._pubchem_get(session, url)
        data = json.loads(content.decode("utf-8"))
        aids = data.get("IdentifierList", {}).get("AID", [])
        results = [{"AID": aid, "type": "assay_id"} for aid in aids]
        logging.info(f"Found {len(aids)} assays for target")
        return results

    async def _query_by_compound(self, session, **kwargs) -> List[Dict]:
        """Get assays where a compound was tested."""
        cid = kwargs.get("cid")
        aids_type = kwargs.get("aids_type", "active")  # all, active, inactive

        if not cid:
            raise ValueError("cid must be provided for compound query")

        # Note: JSON endpoint returns empty, use XML instead
        url = f"{self.base_url}/compound/cid/{cid}/aids/XML?aids_type={aids_type}"
        _, content = await self._pubchem_get(session, url)
        # Parse XML response
        root = ET.fromstring(content.decode("utf-8"))
        ns = {'pug': 'http://pubchem.ncbi.nlm.nih.gov/pug_rest'}
        aids = []
        for aid_elem in root.findall('.//pug:AID', ns):
            aids.append(int(aid_elem.text))
        results = [{"CID": cid, "AID": aid, "activity": aids_type} for aid in aids]
        logging.info(f"Found {len(aids)} assays for CID {cid} ({aids_type})")
        return results

    async def _query_by_assay(self, session, **kwargs) -> List[Dict]:
        """Get compounds from an assay."""
        aid = kwargs.get("aid")
        cids_type = kwargs.get("cids_type", "active")  # all, active, inactive

        if not aid:
            raise ValueError("aid must be provided for assay query")

        # Note: JSON endpoint may return empty, use XML instead
        url = f"{self.base_url}/assay/aid/{aid}/cids/XML?cids_type={cids_type}"
        _, content = await self._pubchem_get(session, url)
        # Parse XML response
        root = ET.fromstring(content.decode("utf-8"))
        ns = {'pug': 'http://pubchem.ncbi.nlm.nih.gov/pug_rest'}
        cids = []
        for cid_elem in root.findall('.//pug:CID', ns):
            cids.append(int(cid_elem.text))
        results = [{"AID": aid, "CID": cid, "activity": cids_type} for cid in cids]
        logging.info(f"Found {len(cids)} compounds ({cids_type}) in assay {aid}")
        return results

    async def _query_by_bioactivity(self, session, **kwargs) -> List[Dict]:
        """Get full bioactivity data for an assay."""
        aid = kwargs.get("aid")
        cid_filter = kwargs.get("cid")  # optional: filter by CID

        if not aid:
            raise ValueError("aid must be provided for bioactivity query")

        url = f"{self.base_url}/assay/aid/{aid}/CSV"
        if cid_filter:
            url += f"?cid={cid_filter}"

        _, content = await self._pubchem_get(session, url)
        csv_data = content.decode("utf-8")
        # Parse CSV
        lines = csv_data.strip().split('\n')
        results = []
        if len(lines) > 0:
            headers = lines[0].split(',')
            for line in lines[1:min(51, len(lines))]:  # Limit to 50 results
                values = line.split(',')
                result = dict(zip(headers, values))
                results.append(result)
        logging.info(f"Retrieved bioactivity data for assay {aid}")
        return results


# TODO: add metadata parser for ChemBL
class ChemBLRequester(DBRequester):
    def __init__(self,
        db_url: str="https://www.ebi.ac.uk/chembl/api/data/molecule?molecule_chembl_id={accession}&format=json",
        timeout: int=30
    ) -> None:
        super().__init__(db_url, timeout)

    def _parse_and_save_outputs(self, accession: str="", content: str="", **kwargs) -> str:
        obj = json.loads(content)
        sdf_file = f"./tmp/chembl_{accession}.sdf"
        with open(sdf_file, "w") as f:
            f.write(obj["molecules"][0]["molecule_structures"]["molfile"])
        molecule = Molecule.from_sdf_file(sdf_file)
        return [molecule], [molecule.save_binary()]


class ChEMBLQueryRequester(Requester):
    """
    Query ChEMBL database for bioactivity data.
    Supports:
    1. Target-based search: Find compounds active against a protein target
    2. Molecule-based search: Get bioactivity profile for a compound
    3. Disease/indication search: Find drugs for a therapeutic indication
    """
    def __init__(self, timeout: int=30) -> None:
        super().__init__()
        self.timeout = timeout
        self.base_url = "https://www.ebi.ac.uk/chembl/api/data"

    def print_usage(self) -> str:
        return "\n".join([
            'ChEMBL database query for bioactivity data.',
            'Inputs:',
            '  {"query_type": "target", "target_name": "EGFR"} - Find compounds for a target',
            '  {"query_type": "target", "uniprot_id": "P00533"} - Query by UniProt ID',
            '  {"query_type": "molecule", "molecule_name": "aspirin"} - Get bioactivity for compound',
            '  {"query_type": "molecule", "smiles": "CC(=O)Oc1ccccc1C(=O)O"} - Query by SMILES',
            '  {"query_type": "molecule", "chembl_id": "CHEMBL25"} - Query by ChEMBL ID',
            '  {"query_type": "indication", "disease": "diabetes"} - Find drugs for disease',
            'Optional filters: standard_type="IC50", standard_value_lte=1000, max_phase=4',
            'Outputs: Dict with bioactivity information'
        ])

    def run(self, query_type: str = "target", **kwargs) -> Tuple[List[Dict], List[str]]:
        return asyncio.run(self.run_async(query_type, **kwargs))

    @RateLimiter(max_calls=5, period=1)
    async def run_async(self, query_type: str = "target", **kwargs) -> Tuple[List[Dict], List[str]]:
        results = []

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            if query_type == "target":
                results = await self._query_by_target(session, **kwargs)
            elif query_type == "molecule":
                results = await self._query_by_molecule(session, **kwargs)
            elif query_type == "indication":
                results = await self._query_by_indication(session, **kwargs)
            else:
                raise ValueError(f"Unknown query_type: {query_type}. Use 'target', 'molecule', or 'indication'")

        return results, [json.dumps(results, indent=2)]

    async def _query_by_target(self, session, **kwargs) -> List[Dict]:
        """Find compounds with activity against a target."""
        target_name = kwargs.get("target_name")
        uniprot_id = kwargs.get("uniprot_id")
        target_chembl_id = kwargs.get("target_chembl_id")
        standard_type = kwargs.get("standard_type", "IC50")
        standard_value_lte = kwargs.get("standard_value_lte", 10000)  # nM
        limit = kwargs.get("limit", 50)

        # Step 1: Get target ChEMBL ID if not provided
        if not target_chembl_id:
            if uniprot_id:
                url = f"{self.base_url}/target/search.json?q={uniprot_id}"
            elif target_name:
                url = f"{self.base_url}/target/search.json?q={quote(target_name)}"
            else:
                raise ValueError("Provide target_name, uniprot_id, or target_chembl_id")

            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    data = json.loads(content.decode("utf-8"))
                    targets = data.get("targets", [])
                    if targets:
                        # Prefer single protein targets with high confidence
                        for t in targets:
                            if t.get("target_type") == "SINGLE PROTEIN":
                                target_chembl_id = t.get("target_chembl_id")
                                break
                        if not target_chembl_id and targets:
                            target_chembl_id = targets[0].get("target_chembl_id")
                    logging.info(f"Found target: {target_chembl_id}")
                else:
                    logging.warning(f"Target search failed, status {response.status}")
                    return []

        if not target_chembl_id:
            logging.warning("No target found in ChEMBL")
            return []

        # Step 2: Get activities for target
        url = f"{self.base_url}/activity.json?target_chembl_id={target_chembl_id}&standard_type={standard_type}&standard_value__lte={standard_value_lte}&limit={limit}"

        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                data = json.loads(content.decode("utf-8"))
                activities = data.get("activities", [])

                results = []
                for act in activities:
                    result = {
                        "molecule_chembl_id": act.get("molecule_chembl_id"),
                        "molecule_name": act.get("molecule_pref_name"),
                        "target_chembl_id": act.get("target_chembl_id"),
                        "target_name": act.get("target_pref_name"),
                        "standard_type": act.get("standard_type"),
                        "standard_value": act.get("standard_value"),
                        "standard_units": act.get("standard_units"),
                        "pchembl_value": act.get("pchembl_value"),
                        "assay_chembl_id": act.get("assay_chembl_id"),
                        "assay_description": act.get("assay_description", "")[:100] if act.get("assay_description") else None,
                    }
                    results.append(result)
                logging.info(f"Found {len(results)} activities for target {target_chembl_id}")
                return results
            else:
                logging.warning(f"Activity query failed, status {response.status}")
                return []

    async def _query_by_molecule(self, session, **kwargs) -> List[Dict]:
        """Get bioactivity profile for a molecule."""
        molecule_name = kwargs.get("molecule_name")
        smiles = kwargs.get("smiles")
        chembl_id = kwargs.get("chembl_id")
        limit = kwargs.get("limit", 50)

        # Step 1: Get molecule ChEMBL ID if not provided
        if not chembl_id:
            if molecule_name:
                url = f"{self.base_url}/molecule/search.json?q={quote(molecule_name)}"
            elif smiles:
                # Use filter endpoint with exact SMILES match (more reliable than flexmatch)
                url = f"{self.base_url}/molecule/filter.json?molecule_structures__canonical_smiles__exact={quote(smiles)}"
            else:
                raise ValueError("Provide molecule_name, smiles, or chembl_id")

            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    data = json.loads(content.decode("utf-8"))
                    # Search endpoint returns {"molecules": [...]}, filter returns molecule directly
                    if molecule_name:
                        molecules = data.get("molecules", [])
                        if molecules:
                            chembl_id = molecules[0].get("molecule_chembl_id")
                    else:
                        # Filter endpoint returns molecule object directly
                        chembl_id = data.get("molecule_chembl_id")
                    logging.info(f"Found molecule: {chembl_id}")
                else:
                    logging.warning(f"Molecule search failed, status {response.status}")
                    return []

        if not chembl_id:
            logging.warning("No molecule found in ChEMBL")
            return []

        # Step 2: Get activities for molecule
        url = f"{self.base_url}/activity.json?molecule_chembl_id={chembl_id}&limit={limit}"

        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                data = json.loads(content.decode("utf-8"))
                activities = data.get("activities", [])

                results = []
                for act in activities:
                    result = {
                        "molecule_chembl_id": act.get("molecule_chembl_id"),
                        "molecule_name": act.get("molecule_pref_name"),
                        "target_chembl_id": act.get("target_chembl_id"),
                        "target_name": act.get("target_pref_name"),
                        "target_organism": act.get("target_organism"),
                        "standard_type": act.get("standard_type"),
                        "standard_value": act.get("standard_value"),
                        "standard_units": act.get("standard_units"),
                        "pchembl_value": act.get("pchembl_value"),
                        "assay_chembl_id": act.get("assay_chembl_id"),
                        "assay_description": act.get("assay_description", "")[:100] if act.get("assay_description") else None,
                    }
                    results.append(result)
                logging.info(f"Found {len(results)} activities for molecule {chembl_id}")
                return results
            else:
                logging.warning(f"Activity query failed, status {response.status}")
                return []

    async def _query_by_indication(self, session, **kwargs) -> List[Dict]:
        """Find drugs for a disease/therapeutic indication."""
        disease = kwargs.get("disease")
        max_phase = kwargs.get("max_phase")  # 0-4, where 4 = approved
        limit = kwargs.get("limit", 50)

        if not disease:
            raise ValueError("Provide disease name for indication query")

        # Search drug indications - use simpler search term for better matches
        # ChEMBL uses MeSH headings, so try searching with key terms
        search_term = disease.lower().replace("cancer", "neoplasm").replace("tumor", "neoplasm")
        url = f"{self.base_url}/drug_indication.json?mesh_heading__icontains={quote(search_term)}&limit={limit}"

        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                data = json.loads(content.decode("utf-8"))
                indications = data.get("drug_indications", [])

                results = []
                seen_molecules = set()

                for ind in indications:
                    mol_id = ind.get("molecule_chembl_id")
                    if mol_id in seen_molecules:
                        continue
                    seen_molecules.add(mol_id)

                    # Get phase as integer (API returns strings like "4.0")
                    phase_val = ind.get("max_phase_for_ind", 0)
                    try:
                        phase_int = int(float(phase_val)) if phase_val is not None else 0
                    except (ValueError, TypeError):
                        phase_int = 0

                    # Filter by phase if specified
                    if max_phase is not None and phase_int < max_phase:
                        continue

                    result = {
                        "molecule_chembl_id": mol_id,
                        "molecule_name": ind.get("molecule_pref_name"),
                        "indication": ind.get("mesh_heading"),
                        "max_phase_for_ind": phase_int,
                        "phase_description": self._get_phase_description(phase_int),
                        "drug_indication_id": ind.get("drug_indication_id"),
                    }
                    results.append(result)

                logging.info(f"Found {len(results)} drugs for indication '{disease}'")
                return results
            else:
                logging.warning(f"Indication query failed, status {response.status}")
                return []

    def _get_phase_description(self, phase: int) -> str:
        """Convert phase number to description."""
        phase_map = {
            0: "Preclinical",
            1: "Phase I",
            2: "Phase II",
            3: "Phase III",
            4: "Approved",
        }
        return phase_map.get(phase, "Unknown")

# TODO: add metadata parser for UniProt
class UniProtRequester(DBRequester):
    def __init__(self, 
        timeout: int=30
    ) -> None:
        super().__init__(timeout)

    def print_usage(self) -> str:
        return "\n".join([
            'Usage: Query UniProt for a protein.',
            'Inputs: {"accession": str (a UniProt ID)}',
            "Outputs: Protein (an OpenBioMed Protein object)"
        ])

    def _determine_query_url(self, accession: str="", **kwargs) -> str:
        return f"https://rest.uniprot.org/uniprotkb/{accession}?format=json"

    def _parse_and_save_outputs(self, accession: str="", content: str="", **kwargs) -> str:
        obj = json.loads(content)
        protein = Protein.from_fasta(obj["sequence"]["value"])
        protein.name = f"uniprot_{accession}"
        return [protein], [protein.save_binary()]

class PDBRequester(DBRequester):
    def __init__(self, 
        timeout: int=30
    ) -> None:
        super().__init__(timeout)
        self.requires_async = False

    def print_usage(self) -> str:
        return "\n".join([
            'Usage: Query a PDB structure.',
            'Inputs: {"accession": str (PDB/AlphaFoldDB ID), "mode": "metadata" (extracting the metadata of the pdb accession) or "file_only" (downloading the pdb file)}',
            "Outputs: Protein (an OpenBioMed Protein object) or str (the path to the pdb file)"
        ])

    def _determine_query_url(self, accession: str="", mode: str="metadata", **kwargs) -> str:
        if len(accession) == 4:
            if mode == "metadata":
                return f"https://data.rcsb.org/rest/v1/core/entry/{accession}"
            elif mode == "file_only":
                return f"https://files.rcsb.org/download/{accession}.pdb"
            else:
                raise ValueError(f"Invalid mode: {mode}")
        else:
            # AlphaFoldDB ID
            assert mode == "file_only", "Only file_only mode is supported for AlphaFoldDB ID."
            return f"https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v4.pdb"

    def _parse_and_save_outputs(self, accession: str="", content: str="", mode: str="metadata", **kwargs) -> str:
        if mode == "metadata":
            obj = json.loads(content)
            return [obj], [content]
        elif mode == "file_only":
            pdb_file = f"./tmp/pdb_{accession}.pdb"
            with open(pdb_file, "w") as f:
                f.write(content)
            return [pdb_file], [content]

class WebSearchRequester(Tool):
    def __init__(self, timeout: int=30) -> None:
        self.timeout = timeout

    def print_usage(self) -> str:
        return "\n".join([
            'Usage: Search the web for information.',
            'Inputs: {"query": str (a query string)}',
            "Outputs: str (returned results from the search engine)"
        ])

    async def run_async(self, query: str) -> Tuple[List[str], List[str]]:
        """Async web search using IQS client"""
        logging.info(f"[WebSearchRequester] Searching for: {query}")
        start_time = time.time()

        # Use module-level singleton IQS client
        results = await _iqs_client.search(query, search_size=10)

        elapsed = time.time() - start_time
        logging.info(f"[WebSearchRequester] Got {len(results)} results in {elapsed:.2f}s")

        # Deduplicate by url
        seen_urls = set()
        result_texts = []
        for item in results:
            if item["url"] not in seen_urls:
                if item["text"] is not None:
                    result_texts.append(item["text"])
                else:
                    logging.warning(f"[WebSearchRequester] Skipping result with None text: {item['url']}")
                seen_urls.add(item["url"])

        result = "\n\n\n".join(result_texts) if result_texts else ""
        logging.info(f"[WebSearchRequester] Returning {len(result_texts)} unique results")
        return [result], [result]

    def run(self, query: str) -> Tuple[List[str], List[str]]:
        """Sync wrapper for run_async"""
        import warnings
        warnings.warn("WebSearchRequester.run() is deprecated, use run_async()", DeprecationWarning)
        return asyncio.run(self.run_async(query))


class MMSeqsRequester(Requester):
    def __init__(self, 
        host: str="https://api.colabfold.com/", 
        job_url_suffix: str="",
        timeout: int=30
    ) -> None:
        super().__init__()
        self.host = host
        self.job_url_suffix = job_url_suffix
        self.timeout = timeout

    def run(self, query: str) -> Tuple[List[str], List[str]]:
        return asyncio.run(self.run_async(query))

    async def run_async(self, query: str) -> Tuple[List[str], List[str]]:
        raise NotImplementedError

    @RateLimiter(max_calls=5, period=1)
    async def submit_job(self, data: Dict[str, Any]) -> str:
        content = {"status": "UNKNOWN"}
        while True:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    async with session.post(
                        url=f"{self.host}/ticket{self.job_url_suffix}",
                        data=data,
                    ) as response:
                        if response.status == 200:
                            content = await response.read()
                            content = json.loads(content.decode("utf-8"))
                            if not content["status"] in ["UNKNOWN", "RATELIMIT"]:
                                break
                        else:
                            logging.warning(f"HTTP request failed, status {response.status}")
                            raise Exception()
                await asyncio.sleep(5 + random.randint(0, 5))
            except Exception as e:
                content = None
                logging.error(f"Web request failed. Exception: {e}")
                raise e
        
        if content["status"] == "ERROR":
            raise Exception(f'Web API is giving errors. Please confirm your input is valid. If error persists, please try again an hour later.')

        if content["status"] == "MAINTENANCE":
            raise Exception(f'Web API is undergoing maintenance. Please try again in a few minutes.')

        return content["id"]

    @RateLimiter(max_calls=5, period=1)
    async def wait_finish(self, id: str="") -> str:
        content = {"status": "UNKNOWN"}
        time_elapsed = 0
        while True:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    async with session.get(
                        url=f"{self.host}/ticket/{id}",
                    ) as response:
                        if response.status == 200:
                            content = await response.read()
                            content = json.loads(content.decode("utf-8"))
                            if not content["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                                break
                        else:
                            logging.warning(f"HTTP request failed, status {response.status}")
                            raise Exception()
                t = 5 + random.randint(0, 5)
                time_elapsed += t
                logging.info(f"Current job status: {content['status']}, {time_elapsed} seconds elapsed.")
                await asyncio.sleep(t)
            except Exception as e:
                content = None
                logging.error(f"Web request failed. Exception: {e}")
                raise e
        return content["status"]
    
    @RateLimiter(max_calls=5, period=1)
    async def download(self, id: str="") -> str:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(
                    url=f"{self.host}/result/download/{id}",
                ) as response:
                    if response.status == 200:
                        content = await response.read()
                        return content
                    else:
                        logging.warning(f"HTTP request failed, status {response.status}")
                        raise Exception()
        except Exception as e:
            content = None
            logging.error(f"Web request failed. Exception: {e}")
            raise e

class MSARequester(MMSeqsRequester):
    def __init__(self, 
        host: str="https://api.colabfold.com", 
        mode: str="all",
        timeout: int=30
    ) -> None:
        super().__init__(host=host, job_url_suffix="/msa", timeout=timeout)
        self.mode = mode

    def print_usage(self) -> str:
        return "\n".join([
            'Usage: Perform multiple sequence alignment.',
            'Inputs: {"protein": Protein (an OpenBioMed Protein object)}',
            "Outputs: str (the path to the .a3m file that contains the MSA results)"
        ])

    async def run_async(self, protein: Protein="") -> Tuple[List[str], List[str]]:
        fasta = f">1\n{protein.sequence}\n"
        data = {
            "q": fasta,
            "mode": self.mode,
        }
        while True:
            id = await self.submit_job(data)
            logging.info(f"Request id: {id}")
            status = await self.wait_finish(id)
            if status == "COMPLETE":
                break
        content = await self.download(id)
        timestamp = int(datetime.now().timestamp() * 1000)
        tar_file = f"./tmp/msa_results_{timestamp}.tar.gz"
        with open(tar_file, "wb") as f: f.write(content)
        logging.info(f"File saved at {tar_file}")
        with tarfile.open(tar_file) as tar_gz:
            folder_name = tar_file.rstrip(".tar.gz")
            os.makedirs(folder_name, exist_ok=True)
            tar_gz.extractall(folder_name)
        ret = f"{folder_name}/uniref.a3m"
        return [ret], [ret]

class FoldSeekRequester(MMSeqsRequester):
    def __init__(self, 
        host: str="https://search.foldseek.com/api", 
        mode: str="3diaa",
        database: List[str]=["BFVD", "afdb50", "afdb-swissprot", "afdb-proteome", "bfmd", "cath50", "mgnify_esm30", "pdb100", "gmgcl_id"],
        timeout: int=60
    ) -> None:
        super().__init__(host, "", timeout)
        self.mode = mode
        self.database = database

    def print_usage(self) -> str:
        return "\n".join([
            'Usage: Perform FoldSeek search for similar structures.',
            'Inputs: {"protein": Protein (an OpenBioMed Protein object)}',
            "Outputs: str (the path to the .m8 file that contains the FoldSeek results)"
        ])

    async def run_async(self, protein: Protein="") -> Tuple[List[str], List[str]]:
        timestamp = int(datetime.now().timestamp() * 1000)
        pdb_file = f"./tmp/protein_{timestamp}.pdb"
        protein.save_pdb(pdb_file)
        form_data = aiohttp.FormData()
        form_data.add_field("mode", self.mode)
        for db in self.database:
            form_data.add_field("database[]", db)
        # Add the file field (open file in binary mode)
        f = open(pdb_file, 'rb')
        form_data.add_field('q', f, filename=pdb_file, content_type='application/octet-stream')

        try:
            while True:
                id = await self.submit_job(form_data)
                logging.info(f"Request id: {id}")
                status = await self.wait_finish(id)
                if status == "COMPLETE":
                    logging.info("Task completed. Try downloading...")
                    break
            content = await self.download(id)
        finally:
            f.close()
        tar_file = f"./tmp/foldseek_results_{timestamp}.tar.gz"
        with open(tar_file, "wb") as f: f.write(content)
        logging.info(f"File saved at {tar_file}")
        with tarfile.open(tar_file) as tar_gz:
            folder_name = tar_file.rstrip(".tar.gz")
            os.makedirs(folder_name, exist_ok=True)
            tar_gz.extractall(folder_name)
        ret = f"{folder_name}"
        return [ret], [ret]


class STRINGRequester(Requester):
    """
    Query STRING database for protein-protein interactions.
    Supports:
    1. Query by UniProt ID or gene symbol -> get interaction partners with confidence scores
    2. Configurable confidence threshold (150=low, 400=medium, 700=high, 900=highest)
    """
    def __init__(self, timeout: int=30) -> None:
        super().__init__()
        self.timeout = timeout
        self.base_url = "https://string-db.org/api/json"

    def print_usage(self) -> str:
        return "\n".join([
            'STRING database protein-protein interaction query.',
            'Inputs:',
            '  {"uniprot_id": "P04637"} - Query by UniProt ID',
            '  {"uniprot_id": "P04637", "species": 9606, "required_score": 700, "limit": 50}',
            'Outputs: Dict with interaction partners and confidence scores'
        ])

    def run(self, uniprot_id: str, species: int=9606, required_score: int=700, limit: int=50) -> Tuple[List[Dict], List[str]]:
        return asyncio.run(self.run_async(uniprot_id, species, required_score, limit))

    @RateLimiter(max_calls=5, period=1)
    async def run_async(self, uniprot_id: str, species: int=9606, required_score: int=700, limit: int=50) -> Tuple[List[Dict], List[str]]:
        """
        Query STRING for interaction partners.

        Args:
            uniprot_id: UniProt accession (e.g., P04637 for TP53)
            species: NCBI taxonomy ID (default: 9606 for human)
            required_score: Minimum confidence score (150=low, 400=medium, 700=high, 900=highest)
            limit: Maximum number of interactors to return

        Returns:
            List of interaction records with confidence scores
        """
        url = f"{self.base_url}/interaction_partners"

        params = {
            "identifiers": uniprot_id,
            "species": species,
            "required_score": required_score,
            "limit": limit
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        content = await response.read()
                        data = json.loads(content.decode("utf-8"))
                        logging.info(f"Found {len(data)} interactions for {uniprot_id}")
                    elif response.status == 404:
                        logging.warning(f"No interactions found for {uniprot_id}")
                        return [], []
                    else:
                        logging.warning(f"STRING query failed, status {response.status}")
                        raise Exception(f"STRING API returned status {response.status}")
        except Exception as e:
            logging.error(f"STRING query failed. Exception: {e}")
            raise e

        # Parse and format results
        results = []
        for interaction in data:
            result = {
                "query_protein": interaction.get("preferredName_A"),
                "partner_string_id": interaction.get("stringId_B"),
                "partner_gene": interaction.get("preferredName_B"),
                "combined_score": interaction.get("score"),
                "scores": {
                    "experimental": interaction.get("escore"),
                    "text_mining": interaction.get("tscore"),
                    "database": interaction.get("dscore"),
                    "coexpression": interaction.get("ascore"),
                    "phylogenetic": interaction.get("pscore"),
                    "gene_fusion": interaction.get("fscore"),
                    "neighborhood": interaction.get("nscore")
                },
                "ncbi_taxon_id": interaction.get("ncbiTaxonId")
            }
            results.append(result)

        return results, [json.dumps(results, indent=2)]


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    """
    requester = UniProtRequester()
    asyncio.run(requester.run("P0DTC2"))

    requester = PDBRequester()
    asyncio.run(requester.run("6LVN"))
    
    requester = PubChemRequester()
    asyncio.run(requester.run("240"))

    requester = ChemBLRequester()
    asyncio.run(requester.run("CHEMBL941"))

    requester = MSARequester()
    asyncio.run(requester.run(Protein.from_binary_file("./tmp/uniprot_P0DTC2.pkl")))
    #asyncio.run(requester.run(Protein.from_fasta("MMVEVRFFGPIKEENFFIKANDLKELRAILQEKEGLKEWLGVCAIALNDHLIDNLNTPLKDGDVISLLPPVCGG")))
    requester = FoldSeekRequester(database=["afdb50"])
    asyncio.run(requester.run(Protein.from_pdb_file("./tmp/demo/foldseek.pdb")))

    requester = PDBRequester("https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v4.pdb")
    asyncio.run(requester.run("A0A2E8J446"))

    websearchrequester = WebSearchRequester()
    qurey = "Please tell me something about diabetes"  
    print(websearchrequester.run(qurey))

    requester = PubChemRequester(db_url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{accession}/SDF")
    asyncio.run(requester.run("dimethoxy-sulfanylidene-(3,5,6-trichloropyridin-2-yl)oxy-lambda5-phosphane"))
    """

    # requester = PubChemStructureRequester()
    # asyncio.run(requester.run(Molecule.from_smiles("CN1CCC[C@H]1COC2=NC3=C(CCN(C3)C4=CC=CC5=C4C(=CC=C5)Cl)C(=N2)N6CCN([C@H](C6)CC#N)C(=O)C(=C)F"), threshold=0.8))

    requester = PDBRequester(db_url="https://data.rcsb.org/rest/v1/core/entry/{accession}")
    requester.run("4xli")