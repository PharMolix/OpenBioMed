import json
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ClinicalTrialsAPI:
    BASE_URL = "https://clinicaltrials.gov/api/v2"

    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"accept": "application/json"})
        retry_kwargs = {
            "total": max_retries,
            "backoff_factor": 0.5,
            "status_forcelist": [429, 500, 502, 503, 504],
        }
        try:
            retry = Retry(allowed_methods=["GET"], **retry_kwargs)
        except TypeError:
            retry = Retry(method_whitelist=["GET"], **retry_kwargs)
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _flatten_params(self, prefix: str, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not data:
            return {}
        return {f"{prefix}.{key}": value for key, value in data.items()}

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = self.session.get(
            f"{self.BASE_URL}/{endpoint.lstrip('/')}",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_studies(
        self,
        query: Optional[Dict[str, Any]] = None,
        filter: Optional[Dict[str, Any]] = None,
        post_filter: Optional[Dict[str, Any]] = None,
        fields: Optional[List[str]] = None,
        sort: Optional[List[str]] = None,
        page_size: int = 50,
        page_token: Optional[str] = None,
        count_total: bool = False,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "format": "json",
            "markupFormat": "markdown",
            "pageSize": str(page_size),
            "countTotal": str(count_total).lower(),
        }
        if page_token:
            params["pageToken"] = page_token
        if fields:
            params["fields"] = ",".join(fields)
        if sort:
            params["sort"] = ",".join(sort)

        params.update(self._flatten_params("query", query))
        params.update(self._flatten_params("filter", filter))
        params.update(self._flatten_params("postFilter", post_filter))
        return self._get("studies", params=params)

    def get_study(self, nct_id: str, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"format": "json", "markupFormat": "markdown"}
        if fields:
            params["fields"] = ",".join(fields)
        return self._get(f"studies/{nct_id}", params=params)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Local ClinicalTrials adapter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    get_studies = subparsers.add_parser("get_studies")
    get_studies.add_argument("--query-cond")
    get_studies.add_argument("--query-term")
    get_studies.add_argument("--filter-overall-status")
    get_studies.add_argument("--fields", nargs="*")
    get_studies.add_argument("--sort", nargs="*")
    get_studies.add_argument("--page-size", type=int, default=20)
    get_studies.add_argument("--count-total", action="store_true")

    get_study = subparsers.add_parser("get_study")
    get_study.add_argument("nct_id")
    get_study.add_argument("--fields", nargs="*")

    args = parser.parse_args()
    api = ClinicalTrialsAPI()

    if args.command == "get_studies":
        query = {}
        if args.query_cond:
            query["cond"] = args.query_cond
        if args.query_term:
            query["term"] = args.query_term

        filter_params = {}
        if args.filter_overall_status:
            filter_params["overallStatus"] = args.filter_overall_status

        result = api.get_studies(
            query=query or None,
            filter=filter_params or None,
            fields=args.fields,
            sort=args.sort,
            page_size=args.page_size,
            count_total=args.count_total,
        )
    else:
        result = api.get_study(args.nct_id, fields=args.fields)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
