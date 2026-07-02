"""
Test script for ScanpyAnalysis tool.
Tests the tool functionality using scanpy's built-in datasets.

Usage:
    python test/test_scanpy_analysis_tool.py
    python test/test_scanpy_analysis_tool.py --api --url http://127.0.0.1:8095
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import scanpy as sc
from open_biomed.tools.scanpy_analysis_tool import ScanpyAnalysis


def test_tool_with_builtin_data():
    """
    Test the ScanpyAnalysis tool by:
    1. Loading scanpy's built-in pbmc3k dataset
    2. Running full pipeline
    3. Verifying the output structure
    """
    print("=" * 60)
    print("Testing ScanpyAnalysis Tool")
    print("=" * 60)

    # Create output directory
    output_dir = "./tmp"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    # Step 1: Load built-in pbmc3k dataset directly from scanpy
    print("\n[1] Loading scanpy built-in pbmc3k dataset...")
    try:
        adata = sc.datasets.pbmc3k()
        print(f"    Dataset loaded: {adata.n_obs} cells, {adata.n_vars} genes")
        # Save to temp file for testing
        input_file = os.path.join(output_dir, "test_pbmc3k_input.h5ad")
        adata.write_h5ad(input_file)
        print(f"    Saved input to: {input_file}")
    except Exception as e:
        print(f"    ERROR: Failed to load dataset: {e}")
        return False

    # Step 2: Test load operation
    print("\n[2] Testing load operation...")
    try:
        tool = ScanpyAnalysis()
        result, msg = tool.run(input_file, operation="load", output_dir=output_dir)
        print(f"    Message: {msg}")
        print(f"    n_obs: {result['n_obs']}, n_vars: {result['n_vars']}")
        print(f"    format: {result['format']}")
    except Exception as e:
        print(f"    ERROR: Load operation failed: {e}")
        return False

    # Step 3: Test QC operation
    print("\n[3] Testing qc operation...")
    try:
        result, msg = tool.run(input_file, operation="qc", output_dir=output_dir)
        print(f"    Message: {msg}")
        print(f"    Pre-filter: {result['pre_filter_cells']} cells, {result['pre_filter_genes']} genes")
        print(f"    Post-filter: {result['post_filter_cells']} cells, {result['post_filter_genes']} genes")
        print(f"    Cells removed: {result['cells_removed']}")
    except Exception as e:
        print(f"    ERROR: QC operation failed: {e}")
        return False

    # Step 4: Test normalize operation on filtered data
    print("\n[4] Testing normalize operation...")
    try:
        filtered_file = result['output_file']
        result, msg = tool.run(filtered_file, operation="normalize", output_dir=output_dir)
        print(f"    Message: {msg}")
        print(f"    n_hvgs_detected: {result['n_hvgs_detected']}")
    except Exception as e:
        print(f"    ERROR: Normalize operation failed: {e}")
        return False

    # Step 5: Test cluster operation on normalized data
    print("\n[5] Testing cluster operation...")
    try:
        normalized_file = result['output_file']
        result, msg = tool.run(normalized_file, operation="cluster", output_dir=output_dir, resolution=0.5)
        print(f"    Message: {msg}")
        print(f"    n_clusters: {result['n_clusters']}")
        print(f"    Cluster sizes: {result['cluster_sizes']}")
    except Exception as e:
        print(f"    ERROR: Cluster operation failed: {e}")
        return False

    # Step 6: Test markers operation on clustered data
    print("\n[6] Testing markers operation...")
    try:
        clustered_file = result['output_file']
        result, msg = tool.run(clustered_file, operation="markers", output_dir=output_dir)
        print(f"    Message: {msg}")
        print(f"    n_groups: {result['n_groups']}")
        print(f"    Markers CSV: {result['markers_csv']}")
        # Print top markers for first cluster
        if 'top_markers' in result and '0' in result['top_markers']:
            print(f"    Top markers for cluster 0: {result['top_markers']['0']['genes'][:5]}")
    except Exception as e:
        print(f"    ERROR: Markers operation failed: {e}")
        return False

    # Step 7: Test full pipeline
    print("\n[7] Testing full_pipeline operation...")
    try:
        result, msg = tool.run(input_file, operation="full_pipeline", output_dir=output_dir, resolution=0.5)
        print(f"    Message: {msg}")
        print(f"    Pipeline steps: {result['pipeline_steps']}")
        print(f"    QC stats: cells_removed={result['qc_stats']['cells_removed']}")
        print(f"    Normalize stats: n_hvgs={result['normalize_stats']['n_hvgs']}")
        print(f"    Cluster stats: n_clusters={result['cluster_stats']['n_clusters']}")
        print(f"    Final output: {result['output_file']}")
    except Exception as e:
        print(f"    ERROR: Full pipeline failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    return True


def test_tool_error_handling():
    """
    Test error handling for invalid inputs.
    """
    print("\n" + "=" * 60)
    print("Testing Error Handling")
    print("=" * 60)

    tool = ScanpyAnalysis()

    # Test 1: Non-existent file
    print("\n[1] Testing non-existent file...")
    try:
        result, msg = tool.run("/nonexistent/path/file.h5ad", operation="load")
        print(f"    ERROR: Should have raised FileNotFoundError")
        return False
    except FileNotFoundError as e:
        print(f"    Correctly raised FileNotFoundError: {e}")
    except Exception as e:
        print(f"    ERROR: Unexpected exception: {e}")
        return False

    # Test 2: Unsupported operation
    print("\n[2] Testing unsupported operation...")
    try:
        with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as f:
            # Create minimal h5ad file
            import anndata as ad
            import numpy as np
            adata = ad.AnnData(np.random.rand(10, 5))
            adata.write_h5ad(f.name)
            temp_file = f.name

        result, msg = tool.run(temp_file, operation="invalid_operation")
        print(f"    ERROR: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"    Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"    ERROR: Unexpected exception: {e}")
        return False

    # Test 3: Unsupported file format
    print("\n[3] Testing unsupported file format...")
    try:
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"some text")
            temp_file = f.name

        result, msg = tool.run(temp_file, operation="load")
        print(f"    ERROR: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"    Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"    ERROR: Unexpected exception: {e}")
        return False

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

    # Check if server is running
    base_url = os.environ.get("OPENBIOMED_API_BASE_URL", "http://127.0.0.1:8095")

    print(f"\n[1] Checking server health at {base_url}...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"    Server is healthy: {response.json()}")
        else:
            print(f"    Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"    ERROR: Server not running at {base_url}")
        print("    Start server with: python -m uvicorn open_biomed.scripts.run_server:app --host 0.0.0.0 --port 8095")
        return False

    # Prepare test data
    print("\n[2] Preparing test data...")
    output_dir = "./tmp"
    os.makedirs(output_dir, exist_ok=True)
    adata = sc.datasets.pbmc3k()
    input_file = os.path.join(output_dir, "api_test_pbmc3k.h5ad")
    adata.write_h5ad(input_file)
    print(f"    Saved test data to: {input_file}")

    # Test load operation via API
    print("\n[3] Testing scanpy_analysis via API (load operation)...")
    try:
        response = requests.post(
            f"{base_url}/run_pipeline/",
            json={
                "task": "scanpy_analysis",
                "protein": input_file,
                "query": "load"
            },
            timeout=30
        )
        print(f"    Response status: {response.status_code}")
        result = response.json()
        print(f"    Response: n_obs={result.get('n_obs')}, n_vars={result.get('n_vars')}")
        print(f"    Message: {result.get('message', result.get('description'))}")
        if response.status_code == 200:
            print("    Load operation successful via API")
        else:
            print(f"    WARNING: Unexpected response: {result}")
    except Exception as e:
        print(f"    ERROR: Request failed: {e}")
        return False

    # Test full_pipeline via API
    print("\n[4] Testing scanpy_analysis via API (full_pipeline)...")
    try:
        response = requests.post(
            f"{base_url}/run_pipeline/",
            json={
                "task": "scanpy_analysis",
                "protein": input_file,
                "query": "full_pipeline",
                "similarity": 0.5,  # resolution
                "num_rounds": 200,  # min_genes
                "population_size": 3,  # min_cells
                "diversity_weight": 5.0,  # max_mt_percent
                "max_mutations": 2000,  # n_top_genes
                "required_score": 10,  # n_neighbors
                "limit": 40  # n_pcs
            },
            timeout=120  # Full pipeline takes longer
        )
        print(f"    Response status: {response.status_code}")
        result = response.json()
        if response.status_code == 200:
            print(f"    Pipeline steps: {result.get('pipeline_steps')}")
            print(f"    n_clusters: {result.get('cluster_stats', {}).get('n_clusters')}")
            print(f"    Output file: {result.get('output_file')}")
            print("    Full pipeline successful via API")
        else:
            print(f"    WARNING: Unexpected response: {result}")
    except Exception as e:
        print(f"    ERROR: Request failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("API endpoint tests completed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test ScanpyAnalysis tool")
    parser.add_argument("--api", action="store_true", help="Test API endpoint (requires running server)")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8095", help="API base URL")
    args = parser.parse_args()

    if args.url:
        os.environ["OPENBIOMED_API_BASE_URL"] = args.url

    # Run tests
    success = True
    success = test_tool_with_builtin_data() and success
    success = test_tool_error_handling() and success

    if args.api:
        success = test_api_endpoint() and success

    sys.exit(0 if success else 1)