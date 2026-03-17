import json
import os
from typing import Any, Dict


class SearchAPI:
    def __init__(self, api_key: str | None = None, max_results: int = 5):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.max_results = max_results

    def run(self, query: str) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY is not set.")

        try:
            from langchain_tavily import TavilySearch
        except ImportError as exc:
            raise RuntimeError("langchain_tavily is not installed.") from exc

        tool = TavilySearch(
            tavily_api_key=self.api_key,
            max_results=self.max_results,
            topic="general",
            include_answer=True,
        )
        return tool.invoke(query)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Local Tavily adapter")
    parser.add_argument("query")
    parser.add_argument("--max-results", type=int, default=5)
    args = parser.parse_args()

    api = SearchAPI(max_results=args.max_results)
    result = api.run(args.query)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
