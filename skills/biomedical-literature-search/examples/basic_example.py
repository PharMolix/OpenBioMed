"""
Biomedical Literature Search Examples

This script demonstrates how to search PubMed and fetch bioRxiv papers
using direct HTTP requests to their APIs.
"""

import requests
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from urllib.parse import quote


def search_pubmed(query: str, max_results: int = 10) -> list:
    """
    Search PubMed using NCBI E-utilities API.

    Args:
        query: Search keywords (e.g., "PD-1 inhibitor cancer")
        max_results: Maximum number of results to return

    Returns:
        List of paper dictionaries with title, abstract, authors, etc.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # Step 1: Search for PMIDs
    search_url = f"{base_url}/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json"
    }

    response = requests.get(search_url, params=search_params)
    search_data = response.json()

    pmids = search_data["esearchresult"]["idlist"]

    if not pmids:
        return []

    # Step 2: Fetch paper details
    fetch_url = f"{base_url}/efetch.fcgi"
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml"
    }

    response = requests.get(fetch_url, params=fetch_params)
    root = ET.fromstring(response.content)

    papers = []
    for article in root.findall(".//PubmedArticle"):
        # Extract title
        title_elem = article.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else "N/A"

        # Extract abstract
        abstract_elem = article.find(".//Abstract/AbstractText")
        abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"

        # Extract authors
        authors = []
        for author in article.findall(".//Author"):
            lastname = author.find("LastName")
            forename = author.find("ForeName")
            if lastname is not None:
                name = lastname.text
                if forename is not None:
                    name = f"{name} {forename.text}"
                authors.append(name)

        # Extract DOI
        doi = "N/A"
        for article_id in article.findall(".//ArticleId"):
            if article_id.get("IdType") == "doi":
                doi = article_id.text
                break

        # Extract PMID
        pmid_elem = article.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else "N/A"

        # Extract date
        date = "N/A"
        pub_date = article.find(".//PubDate")
        if pub_date is not None:
            year = pub_date.find("Year")
            month = pub_date.find("Month")
            day = pub_date.find("Day")
            if year is not None:
                date = year.text
                if month is not None:
                    date = f"{date}-{month.text}"
                if day is not None:
                    date = f"{date}-{day.text}"

        papers.append({
            "title": title,
            "authors": ", ".join(authors) if authors else "N/A",
            "abstract": abstract,
            "doi": doi,
            "pmid": pmid,
            "date": date,
            "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        })

    return papers


def fetch_biorxiv_papers(
    start_date: str = None,
    end_date: str = None,
    days: int = 30,
    category: str = None
) -> list:
    """
    Fetch papers from bioRxiv by date range.

    Args:
        start_date: Start date (YYYY-MM-DD), defaults to 30 days ago
        end_date: End date (YYYY-MM-DD), defaults to today
        days: Number of recent days to fetch (alternative to date range)
        category: Filter by subject category (e.g., "cancer_biology")

    Returns:
        List of paper dictionaries with title, abstract, authors, etc.
    """
    # Build date range
    if not start_date or not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Build URL
    url = f"https://api.biorxiv.org/details/biorxiv/{start_date}/{end_date}"

    if category:
        url += f"?category={category}"

    # Fetch data
    response = requests.get(url)
    data = response.json()

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

    return papers


def format_paper_summary(paper: dict, source: str = "unknown") -> str:
    """Format a paper dict into readable text."""
    abstract_preview = paper.get('abstract', 'No abstract available')
    if len(abstract_preview) > 500:
        abstract_preview = abstract_preview[:500] + "..."

    return f"""
[{source.upper()}] {paper.get('title', 'N/A')}
Authors: {paper.get('authors', 'N/A')}
Date: {paper.get('date', 'N/A')}
DOI: {paper.get('doi', 'N/A')}
Link: {paper.get('link', 'N/A')}

Abstract:
{abstract_preview}
"""


def main():
    """Run example literature search."""

    print("=" * 60)
    print("Biomedical Literature Search Example")
    print("=" * 60)

    # Example 1: PubMed search
    print("\n### PubMed Search: PD-1 inhibitor cancer ###\n")

    pubmed_papers = search_pubmed("PD-1 inhibitor cancer", max_results=5)

    print(f"Found {len(pubmed_papers)} papers\n")

    for i, paper in enumerate(pubmed_papers[:3]):
        print(f"--- Paper {i+1} ---")
        print(format_paper_summary(paper, source="pubmed"))

    # Example 2: bioRxiv fetch by date range
    print("\n### bioRxiv Recent Papers (Last 30 days) ###\n")

    biorxiv_papers = fetch_biorxiv_papers(days=30)

    print(f"Total papers found: {len(biorxiv_papers)}\n")

    # Display first 3 papers
    for i, paper in enumerate(biorxiv_papers[:3]):
        print(f"--- Paper {i+1} ---")
        print(format_paper_summary(paper, source="biorxiv"))

    # Example 3: bioRxiv by category
    print("\n### bioRxiv Immunology Category (Last 30 days) ###\n")

    immuno_papers = fetch_biorxiv_papers(days=30, category="immunology")

    if immuno_papers:
        print(f"Immunology papers: {len(immuno_papers)}")
        print(format_paper_summary(immuno_papers[0], source="biorxiv"))
    else:
        print("No papers found in this category for the date range.")


if __name__ == "__main__":
    main()
