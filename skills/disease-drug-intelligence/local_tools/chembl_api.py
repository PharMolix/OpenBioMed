import json
from typing import Any, Dict, Optional

import requests


class ChemblAPI:
    BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = self.session.get(
            f"{self.BASE_URL}/{endpoint.lstrip('/')}",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def search_target(self, query_str: str, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
        return self._get("target/search", {"q": query_str, "limit": limit, "offset": offset})

    def search_molecule(self, query_str: str, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
        return self._get("molecule/search", {"q": query_str, "limit": limit, "offset": offset})

    def get_drug_by_id(self, chembl_id: str) -> Dict[str, Any]:
        return self._get(f"drug/{chembl_id}")

    def get_molecule_by_id(self, chembl_id: str) -> Dict[str, Any]:
        return self._get(f"molecule/{chembl_id}")

    def get_target_by_id(self, chembl_id: str) -> Dict[str, Any]:
        return self._get(f"target/{chembl_id}")

    def get_mechanism(self, molecule_chembl_id: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if molecule_chembl_id:
            params["molecule_chembl_id"] = molecule_chembl_id
        return self._get("mechanism", params)

    def get_drug_indication(
        self,
        molecule_chembl_id: Optional[str] = None,
        efo_term: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if molecule_chembl_id:
            params["molecule_chembl_id"] = molecule_chembl_id
        if efo_term:
            params["efo_term"] = efo_term
        return self._get("drug_indication", params)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Local ChEMBL adapter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_target = subparsers.add_parser("search_target")
    search_target.add_argument("query")
    search_target.add_argument("--limit", type=int, default=10)

    search_molecule = subparsers.add_parser("search_molecule")
    search_molecule.add_argument("query")
    search_molecule.add_argument("--limit", type=int, default=10)

    get_drug = subparsers.add_parser("get_drug_by_id")
    get_drug.add_argument("chembl_id")

    get_molecule = subparsers.add_parser("get_molecule_by_id")
    get_molecule.add_argument("chembl_id")

    get_target = subparsers.add_parser("get_target_by_id")
    get_target.add_argument("chembl_id")

    get_mechanism = subparsers.add_parser("get_mechanism")
    get_mechanism.add_argument("--molecule-chembl-id")
    get_mechanism.add_argument("--limit", type=int, default=20)

    get_indication = subparsers.add_parser("get_drug_indication")
    get_indication.add_argument("--molecule-chembl-id")
    get_indication.add_argument("--efo-term")
    get_indication.add_argument("--limit", type=int, default=20)

    args = parser.parse_args()
    api = ChemblAPI()

    if args.command == "search_target":
        result = api.search_target(args.query, limit=args.limit)
    elif args.command == "search_molecule":
        result = api.search_molecule(args.query, limit=args.limit)
    elif args.command == "get_drug_by_id":
        result = api.get_drug_by_id(args.chembl_id)
    elif args.command == "get_molecule_by_id":
        result = api.get_molecule_by_id(args.chembl_id)
    elif args.command == "get_target_by_id":
        result = api.get_target_by_id(args.chembl_id)
    elif args.command == "get_mechanism":
        result = api.get_mechanism(
            molecule_chembl_id=args.molecule_chembl_id,
            limit=args.limit,
        )
    else:
        result = api.get_drug_indication(
            molecule_chembl_id=args.molecule_chembl_id,
            efo_term=args.efo_term,
            limit=args.limit,
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
