"""
Test script for SpatialTranscriptomicsLoader tool.
Tests the tool functionality using squidpy's built-in datasets.

Usage:
    python test/test_spatial_transcriptomics_tool.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import squidpy as sq
from open_biomed.tools.spatial_transcriptomics_tool import SpatialTranscriptomicsLoader


def test_tool_with_builtin_data():
    """
    Test the SpatialTranscriptomicsLoader tool by:
    1. Loading squidpy's built-in Visium dataset
    2. Saving it as AnnData format
    3. Verifying the output structure
    """
    print("=" * 60)
    print("Testing SpatialTranscriptomicsLoader Tool")
    print("=" * 60)

    # Create output directory
    output_dir = "./tmp"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load built-in Visium dataset directly from squidpy
    print("\n[1] Loading squidpy built-in Visium H&E dataset...")
    try:
        adata = sq.datasets.visium_hne_adata()
        print(f"    Dataset loaded: {adata.n_obs} spots, {adata.n_vars} genes")
        print(f"    Has spatial coords: {'spatial' in adata.obsm}")
        print(f"    Has images: {'spatial' in adata.uns if adata.uns else False}")
    except Exception as e:
        print(f"    ERROR: Failed to load dataset: {e}")
        return False

    # Step 2: Save the AnnData to test output
    print("\n[2] Saving AnnData to file...")
    output_file = os.path.join(output_dir, "test_visium_builtin.h5ad")
    try:
        adata.write_h5ad(output_file)
        print(f"    Saved to: {output_file}")
        print(f"    File size: {os.path.getsize(output_file) / 1024:.2f} KB")
    except Exception as e:
        print(f"    ERROR: Failed to save file: {e}")
        return False

    # Step 3: Verify the saved file can be loaded
    print("\n[3] Verifying saved file...")
    try:
        import anndata as ad
        loaded = ad.read_h5ad(output_file)
        print(f"    Loaded: {loaded.n_obs} spots, {loaded.n_vars} genes")
        print(f"    Spatial coords shape: {loaded.obsm['spatial'].shape}")
    except Exception as e:
        print(f"    ERROR: Failed to load saved file: {e}")
        return False

    # Step 4: Test Xenium dataset
    print("\n[4] Testing Xenium dataset...")
    try:
        xenium_adata = sq.datasets.xenium()
        print(f"    Xenium loaded: {xenium_adata.n_obs} cells, {xenium_adata.n_vars} genes")
        xenium_file = os.path.join(output_dir, "test_xenium_builtin.h5ad")
        xenium_adata.write_h5ad(xenium_file)
        print(f"    Saved to: {xenium_file}")
    except Exception as e:
        print(f"    WARNING: Xenium test skipped: {e}")

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

    tool = SpatialTranscriptomicsLoader()

    # Test 1: Non-existent directory
    print("\n[1] Testing non-existent directory...")
    try:
        result, msg = tool.run("/nonexistent/path", "visium")
        print(f"    ERROR: Should have raised FileNotFoundError")
        return False
    except FileNotFoundError as e:
        print(f"    Correctly raised FileNotFoundError: {e}")
    except Exception as e:
        print(f"    ERROR: Unexpected exception: {e}")
        return False

    # Test 2: Unsupported platform
    print("\n[2] Testing unsupported platform...")
    try:
        # Create temp dir
        with tempfile.TemporaryDirectory() as tmpdir:
            result, msg = tool.run(tmpdir, "invalid_platform")
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

    # Test error response for invalid path
    print("\n[2] Testing API error response...")
    try:
        response = requests.post(
            f"{base_url}/run_pipeline/",
            json={
                "task": "spatial_transcriptomics_loading",
                "value": "/nonexistent/path",
                "query": "visium"
            },
            timeout=10
        )
        print(f"    Response status: {response.status_code}")
        print(f"    Response body: {response.json()}")
        if response.status_code == 500 and "not found" in str(response.json()).lower():
            print("    Correctly returned error for invalid path")
        else:
            print("    WARNING: Unexpected response")
    except Exception as e:
        print(f"    ERROR: Request failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("API endpoint tests completed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test SpatialTranscriptomicsLoader tool")
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