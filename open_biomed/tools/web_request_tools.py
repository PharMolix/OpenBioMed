from abc import abstractmethod, ABC
from typing import Any, Dict, List, Optional, Tuple
import os
import sys
import requests
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import aiohttp
import asyncio
from datetime import datetime
import json
import logging
import random
from ratelimiter import RateLimiter
import tarfile
from urllib.parse import quote
import xml.etree.ElementTree as ET

from open_biomed.data import Molecule, Protein
from open_biomed.tools.base_tool import Tool

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

    @RateLimiter(max_calls=5, period=1)
    async def run_async(self, accession: Any, **kwargs) -> Any:
        url = self._determine_query_url(accession, **kwargs)
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        content = content.decode("utf-8")
                        logging.info("Downloaded results successfully")
                    else:
                        logging.warning(f"HTTP request failed, status {response.status}")
                        raise Exception()
        except Exception as e:
            content = None
            logging.error(f"Download failed. Exception: {e}")
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
        return db_url.format(accession=accession)

    def _parse_and_save_outputs(self, accession: str="", content: str="", **kwargs) -> Tuple[List[Molecule], List[str]]:
        sdf_file = f"./tmp/pubchem_{accession}.sdf"
        with open(sdf_file, "w") as f:
            f.write(content)
        molecule = Molecule.from_sdf_file(sdf_file)
        return [molecule], [molecule.save_binary()]

class PubChemStructureRequester(Requester):
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

    @RateLimiter(max_calls=5, period=1)
    async def run_async(self, molecule: Molecule=None, threshold: float=0.8, max_records=10) -> Tuple[List[Molecule], List[str]]:
        molecule._add_smiles()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                url = self.db_url.format(accession=molecule.smiles, threshold=int(threshold * 100), max_records=max_records)
                async with session.get(url.replace("#", "%23")) as response:
                    if response.status == 200:
                        content = await response.read()
                        content = json.loads(content.decode("utf-8"))
                        logging.info("Downloaded results successfully")
                    elif response.status == 404:
                        logging.info("No similar structures found!")
                        return [molecule], [molecule.save_binary()]
                    else:
                        logging.warning(f"HTTP request failed, status {response.status}")
                        raise Exception()
        except Exception as e:
            content = None
            logging.error(f"Download failed. Exception: {e}")
            raise e
        all_mols, all_files = [], []
        for cid in content['IdentifierList']['CID']:
            mol, mol_file = await self.molecule_requester.run_async(cid)
            all_mols.extend(mol)
            all_files.extend(mol_file)
        return all_mols, all_files

class PubChemBioactivityRequester(Requester):
    """
    Query PubChem for bioactivity data.
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

    @RateLimiter(max_calls=5, period=1)
    async def run_async(self, query_type: str = "compound", **kwargs) -> Tuple[List[Dict], List[str]]:
        results = []

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            if query_type == "target":
                # Get assays targeting a gene
                gene_symbol = kwargs.get("gene_symbol")
                gene_id = kwargs.get("gene_id")

                if gene_symbol:
                    url = f"{self.base_url}/assay/target/genesymbol/{gene_symbol}/aids/JSON"
                elif gene_id:
                    url = f"{self.base_url}/assay/target/geneid/{gene_id}/aids/JSON"
                else:
                    raise ValueError("Either gene_symbol or gene_id must be provided for target query")

                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        data = json.loads(content.decode("utf-8"))
                        aids = data.get("IdentifierList", {}).get("AID", [])
                        results = [{"AID": aid, "type": "assay_id"} for aid in aids]
                        logging.info(f"Found {len(aids)} assays for target")
                    else:
                        logging.warning(f"Target query failed, status {response.status}")

            elif query_type == "compound":
                # Get assays where a compound was tested
                # Note: JSON endpoint returns empty, use XML instead
                cid = kwargs.get("cid")
                aids_type = kwargs.get("aids_type", "active")  # all, active, inactive

                if not cid:
                    raise ValueError("cid must be provided for compound query")

                url = f"{self.base_url}/compound/cid/{cid}/aids/XML?aids_type={aids_type}"
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        # Parse XML response
                        root = ET.fromstring(content.decode("utf-8"))
                        ns = {'pug': 'http://pubchem.ncbi.nlm.nih.gov/pug_rest'}
                        aids = []
                        for aid_elem in root.findall('.//pug:AID', ns):
                            aids.append(int(aid_elem.text))
                        results = [{"CID": cid, "AID": aid, "activity": aids_type} for aid in aids]
                        logging.info(f"Found {len(aids)} assays for CID {cid} ({aids_type})")
                    else:
                        logging.warning(f"Compound query failed, status {response.status}")

            elif query_type == "assay":
                # Get compounds from an assay
                # Note: JSON endpoint may return empty, use XML instead
                aid = kwargs.get("aid")
                cids_type = kwargs.get("cids_type", "active")  # all, active, inactive

                if not aid:
                    raise ValueError("aid must be provided for assay query")

                url = f"{self.base_url}/assay/aid/{aid}/cids/XML?cids_type={cids_type}"
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        # Parse XML response
                        root = ET.fromstring(content.decode("utf-8"))
                        ns = {'pug': 'http://pubchem.ncbi.nlm.nih.gov/pug_rest'}
                        cids = []
                        for cid_elem in root.findall('.//pug:CID', ns):
                            cids.append(int(cid_elem.text))
                        results = [{"AID": aid, "CID": cid, "activity": cids_type} for cid in cids]
                        logging.info(f"Found {len(cids)} compounds ({cids_type}) in assay {aid}")
                    else:
                        logging.warning(f"Assay query failed, status {response.status}")

            elif query_type == "bioactivity":
                # Get full bioactivity data for an assay
                aid = kwargs.get("aid")
                cid_filter = kwargs.get("cid")  # optional: filter by CID

                if not aid:
                    raise ValueError("aid must be provided for bioactivity query")

                url = f"{self.base_url}/assay/aid/{aid}/CSV"
                if cid_filter:
                    url += f"?cid={cid_filter}"

                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        csv_data = content.decode("utf-8")
                        # Parse CSV
                        lines = csv_data.strip().split('\n')
                        if len(lines) > 0:
                            headers = lines[0].split(',')
                            for line in lines[1:min(51, len(lines))]:  # Limit to 50 results
                                values = line.split(',')
                                result = dict(zip(headers, values))
                                results.append(result)
                        logging.info(f"Retrieved bioactivity data for assay {aid}")
                    else:
                        logging.warning(f"Bioactivity query failed, status {response.status}")

            else:
                raise ValueError(f"Unknown query_type: {query_type}. Use 'target', 'compound', 'assay', or 'bioactivity'")

        return results, [json.dumps(results, indent=2)]


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

    def run(self, query: str) -> Tuple[List[str], List[str]]:

        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer 1234567890'
        }
        query_url = "https://staging.chatdd.pharmolix.com/v2/api/deepinsight/generate_query"
        data = {
            'chat_session_id': "FwAalhadkajhddkaadfwes",
            'action': False,
            'chat_messages': [
                {"role": "user", "content": f'<p>{query}</p><p><br></p>'}
            ]
        }
        response = requests.post(query_url, headers=headers, json=data)
        question = response.json()

        rag_url = "http://101.200.137.30:1112/rag/v1/common_rag"

        if not question['query_list']:
            question['query_list'] = []
        question["recall_params"] = {
            "PaperDB": [
                "meeting",
                "pubmed_abstract",
                "pubmed_full_text"
            ],
            "NewsDB": [
                "press",
                "media",
                "wechat",
                "wechat_realtime",
                "press_realtime"
            ],
            "WebSearch": None,
            "Clinicaltrial_DB": [
                "clinicaltrials"
            ],
            "Policy_DB": [
                "policy"
            ],
            "Principle_DB": [
                "principle"
            ],
            "PatentLaw_DB": [
                "patentlaw"
            ]
        }
        question["top_k"] = 5
        res = requests.post(rag_url, json=question)
        #result = {"query": [query] + question['query_list'],
        #          "result": res.json()["data"]}
        #result = {"result": [i["text"] for i in res.json()["data"]]}
        result = "\n\n\n".join([i["text"] for i in res.json()["data"]])
        return [result], [result]


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