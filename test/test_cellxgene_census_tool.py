"""
Test script for CellxGeneCensusQuery tool.
Tests the tool functionality using the CZ CELLxGENE Census API.

Usage:
    python test/test_cellxgene_census_tool.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_biomed.tools.cellxgene_census_tool import CellxGeneCensusQuery


def test_tool_get_summary():
    """Test get_summary operation."""
    print("=" * 60)
    print("Testing CellxGeneCensusQuery Tool - get_summary")
    print("=" * 60)

    tool = CellxGeneCensusQuery()

    print("\n[1] Testing get_summary...")
    try:
        result, msg = tool.run(operation="get_summary")
        print(f"    Message: {msg}")
        print(f"    Census version: {result.get('census_version')}")
        print(f"    Total cells: {result.get('total_cell_count')}")
        print("[PASS] get_summary")
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def test_tool_get_datasets():
    """Test get_datasets operation."""
    print("\n" + "=" * 60)
    print("Testing CellxGeneCensusQuery Tool - get_datasets")
    print("=" * 60)

    tool = CellxGeneCensusQuery()
    output_dir = "./tmp"
    os.makedirs(output_dir, exist_ok=True)

    print("\n[1] Testing get_datasets for homo_sapiens...")
    try:
        result, msg = tool.run(operation="get_datasets", organism="homo_sapiens", output_dir=output_dir)
        print(f"    Message: {msg}")
        print(f"    n_datasets: {result.get('n_datasets')}")
        print(f"    Summary: {result.get('summary')}")
        print("[PASS] get_datasets")
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def test_tool_get_obs():
    """Test get_obs operation."""
    print("\n" + "=" * 60)
    print("Testing CellxGeneCensusQuery Tool - get_obs")
    print("=" * 60)

    tool = CellxGeneCensusQuery()

    print("\n[1] Testing get_obs with tissue filter...")
    try:
        result, msg = tool.run(
            operation="get_obs",
            organism="homo_sapiens",
            obs_value_filter="tissue_general == 'lung' and is_primary_data == True",
            obs_column_names=["cell_type", "tissue_general", "disease"]
        )
        print(f"    Message: {msg}")
        print(f"    n_cells: {result.get('n_cells')}")
        print(f"    unique_counts: {result.get('unique_counts')}")
        print(f"    sample cell types: {result.get('sample_values', {}).get('cell_type', [])[:5]}")
        print("[PASS] get_obs")
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def test_tool_get_var():
    """Test get_var operation."""
    print("\n" + "=" * 60)
    print("Testing CellxGeneCensusQuery Tool - get_var")
    print("=" * 60)

    tool = CellxGeneCensusQuery()

    print("\n[1] Testing get_var with gene filter...")
    try:
        result, msg = tool.run(
            operation="get_var",
            organism="homo_sapiens",
            var_value_filter="feature_name in ['CD4', 'CD8A', 'CD19', 'FOXP3']"
        )
        print(f"    Message: {msg}")
        print(f"    n_genes: {result.get('n_genes')}")
        print(f"    gene_names_sample: {result.get('gene_names_sample')}")
        print("[PASS] get_var")
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def test_tool_get_anndata():
    """Test get_anndata operation with small query."""
    print("\n" + "=" * 60)
    print("Testing CellxGeneCensusQuery Tool - get_anndata")
    print("=" * 60)

    tool = CellxGeneCensusQuery()
    output_dir = "./tmp"
    os.makedirs(output_dir, exist_ok=True)

    print("\n[1] Testing get_anndata with small query (max_cells=1000)...")
    try:
        # Use specific filter to limit cells
        result, msg = tool.run(
            operation="get_anndata",
            organism="homo_sapiens",
            obs_value_filter="cell_type == 'B cell' and tissue_general == 'blood' and is_primary_data == True",
            var_value_filter="feature_name in ['CD19', 'MS4A1', 'CD79A']",
            obs_column_names=["cell_type", "tissue_general", "disease"],
            output_dir=output_dir,
            max_cells=1000
        )
        print(f"    Message: {msg}")
        print(f"    Status: {result.get('status')}")
        if result.get('status') == 'success':
            print(f"    n_cells: {result.get('n_cells')}")
            print(f"    n_genes: {result.get('n_genes')}")
            print(f"    Output file: {result.get('output_file')}")
        else:
            print(f"    n_cells_available: {result.get('n_cells_available')}")
            print(f"    Suggestion: {result.get('suggestion')}")
        print("[PASS] get_anndata")
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def test_tool_error_handling():
    """Test error handling."""
    print("\n" + "=" * 60)
    print("Testing Error Handling")
    print("=" * 60)

    tool = CellxGeneCensusQuery()

    print("\n[1] Testing invalid operation...")
    try:
        result, msg = tool.run(operation="invalid_operation")
        print(f"    ERROR: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"    Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"    ERROR: Unexpected exception: {e}")
        return False

    print("\n[2] Testing invalid organism...")
    try:
        result, msg = tool.run(operation="get_obs", organism="invalid_organism")
        # This may or may not raise an error depending on Census implementation
        print(f"    Result: {result.get('message', 'No error raised')}")
    except Exception as e:
        print(f"    Exception raised (expected): {e}")

    print("\n" + "=" * 60)
    print("Error handling tests passed!")
    print("=" * 60)
    return True


def test_api_endpoint():
    """
    Test the API endpoint by making a request to the server.
    Requires the server to be running.
    """
    import requests

    print("\n" + "=" * 60)
    print("Testing API Endpoint")
    print("=" * 60)

    base_url = os.environ.get("OPENBIOMED_API_BASE_URL", "http://127.0.0.1:8095")

    print(f"\n[1] Checking server health at {base_url}...")
    try:
        response = requests.get(f"{base_url}/healthz", timeout=5)
        if response.status_code == 200:
            print(f"    Server is healthy")
        else:
            print(f"    Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"    ERROR: Server not running at {base_url}")
        return False

    # Test get_summary via API
    print("\n[2] Testing cellxgene_census_query via API (get_summary)...")
    try:
        response = requests.post(
            f"{base_url}/run_pipeline/",
            json={
                "task": "cellxgene_census_query",
                "query": "get_summary"
            },
            timeout=60
        )
        print(f"    Response status: {response.status_code}")
        result = response.json()
        print(f"    Census version: {result.get('census_version')}")
        print(f"    Total cells: {result.get('total_cell_count')}")
        print(f"    Message: {result.get('description', result.get('message'))}")
        if response.status_code == 200:
            print("[PASS] API get_summary")
        else:
            print(f"    WARNING: Unexpected response")
    except Exception as e:
        print(f"    ERROR: Request failed: {e}")
        return False

    # Test get_obs via API
    print("\n[3] Testing cellxgene_census_query via API (get_obs)...")
    try:
        response = requests.post(
            f"{base_url}/run_pipeline/",
            json={
                "task": "cellxgene_census_query",
                "query": "get_obs",
                "text": "homo_sapiens",
                "value": "tissue_general == 'lung' and is_primary_data == True"
            },
            timeout=60
        )
        print(f"    Response status: {response.status_code}")
        result = response.json()
        print(f"    n_cells: {result.get('n_cells')}")
        print(f"    unique_counts: {result.get('unique_counts')}")
        print("[PASS] API get_obs")
    except Exception as e:
        print(f"    ERROR: Request failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("API endpoint tests completed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test CellxGeneCensusQuery tool")
    parser.add_argument("--api", action="store_true", help="Test API endpoint (requires running server)")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8095", help="API base URL")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only (skip get_anndata)")
    args = parser.parse_args()

    if args.url:
        os.environ["OPENBIOMED_API_BASE_URL"] = args.url

    # Run tests
    success = True
    success = test_tool_get_summary() and success
    success = test_tool_get_datasets() and success
    success = test_tool_get_obs() and success
    success = test_tool_get_var() and success

    if not args.quick:
        success = test_tool_get_anndata() and success

    success = test_tool_error_handling() and success

    if args.api:
        success = test_api_endpoint() and success

    print("\n" + "=" * 60)
    if success:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)

    sys.exit(0 if success else 1)