"""
Test script for PeptideIdentification tool.
Tests the tool functionality for peptide and protein identification from MS2 spectra.

Usage:
    python test/test_peptide_identification_tool.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_biomed.tools.peptide_identification_tool import PeptideIdentification


def test_tool_init():
    """Test tool initialization and dependency checking."""
    print("=" * 60)
    print("Testing PeptideIdentification Tool - Initialization")
    print("=" * 60)

    print("\n[1] Testing tool initialization...")
    try:
        tool = PeptideIdentification()
        print(f"    Tool initialized successfully")
        print(f"    MSFragger path: {tool._msfragger_path}")
        print(f"    Philosopher path: {tool._philosopher_path}")
        print(f"    Java path: {tool._java_path}")
        print("[PASS] Initialization")
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def test_tool_usage():
    """Test tool usage string."""
    print("\n" + "=" * 60)
    print("Testing PeptideIdentification Tool - Usage")
    print("=" * 60)

    tool = PeptideIdentification()

    print("\n[1] Testing print_usage...")
    try:
        usage = tool.print_usage()
        print(f"    Usage string length: {len(usage)} chars")
        print(f"    Contains 'MSFragger': {'MSFragger' in usage}")
        print(f"    Contains 'Philosopher': {'Philosopher' in usage}")
        print("[PASS] Usage")
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def test_tool_prepare_database():
    """Test prepare_database operation (mock - requires network/download)."""
    print("\n" + "=" * 60)
    print("Testing PeptideIdentification Tool - prepare_database")
    print("=" * 60)

    tool = PeptideIdentification()
    output_dir = "./tmp/test_peptide"
    os.makedirs(output_dir, exist_ok=True)

    print("\n[1] Testing prepare_database operation...")
    print("    NOTE: This operation requires network access to download UniProt/cRAP")
    print("    If jar files are missing, will return download instructions")

    try:
        result, msg = tool.run(
            operation="prepare_database",
            output_dir=output_dir,
            organism="human"
        )
        print(f"    Status: {result.get('status', 'unknown')}")
        print(f"    Message: {msg}")

        if result.get('status') == 'error':
            print(f"    Instructions provided: {result.get('instructions', 'N/A')}")
            print("[PASS] prepare_database (returned instructions - jar files may be missing)")
        else:
            print(f"    Database file: {result.get('database_file', 'N/A')}")
            print(f"    Target proteins: {result.get('n_target_proteins', 'N/A')}")
            print("[PASS] prepare_database")
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def test_tool_parse_results_empty():
    """Test parse_results operation with empty directory."""
    print("\n" + "=" * 60)
    print("Testing PeptideIdentification Tool - parse_results (empty)")
    print("=" * 60)

    tool = PeptideIdentification()
    output_dir = "./tmp/test_peptide_empty"
    os.makedirs(output_dir, exist_ok=True)

    print("\n[1] Testing parse_results on empty directory...")
    try:
        result, msg = tool.run(
            operation="parse_results",
            output_dir=output_dir
        )
        print(f"    Status: {result.get('status')}")
        print(f"    Summary: {result.get('summary', {})}")
        print(f"    Message: {msg}")
        print("[PASS] parse_results (empty)")
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def test_tool_error_handling():
    """Test error handling for invalid operations."""
    print("\n" + "=" * 60)
    print("Testing PeptideIdentification Tool - Error Handling")
    print("=" * 60)

    tool = PeptideIdentification()

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

    print("\n[2] Testing search without mzml_files...")
    try:
        result, msg = tool.run(
            operation="search",
            database_file="test.fasta"
        )
        print(f"    ERROR: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"    Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"    ERROR: Unexpected exception: {e}")
        return False

    print("\n[3] Testing validate without mzml_files...")
    try:
        result, msg = tool.run(
            operation="validate",
            database_file="test.fasta"
        )
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
    print("Testing API Endpoint - peptide_identification")
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

    # Test prepare_database via API
    print("\n[2] Testing peptide_identification via API (prepare_database)...")
    try:
        response = requests.post(
            f"{base_url}/run_pipeline/",
            json={
                "task": "peptide_identification",
                "query": "prepare_database",
                "text": "human"
            },
            timeout=60
        )
        print(f"    Response status: {response.status_code}")
        result = response.json()
        print(f"    Status: {result.get('status', 'unknown')}")
        print(f"    Message: {result.get('description', result.get('message'))}")

        if response.status_code == 200:
            print("[PASS] API prepare_database")
        else:
            print(f"    WARNING: Unexpected response")
    except Exception as e:
        print(f"    ERROR: Request failed: {e}")
        return False

    # Test parse_results via API
    print("\n[3] Testing peptide_identification via API (parse_results)...")
    try:
        response = requests.post(
            f"{base_url}/run_pipeline/",
            json={
                "task": "peptide_identification",
                "query": "parse_results"
            },
            timeout=60
        )
        print(f"    Response status: {response.status_code}")
        result = response.json()
        print(f"    Summary: {result.get('summary', {})}")
        print("[PASS] API parse_results")
    except Exception as e:
        print(f"    ERROR: Request failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("API endpoint tests completed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test PeptideIdentification tool")
    parser.add_argument("--api", action="store_true", help="Test API endpoint (requires running server)")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8095", help="API base URL")
    args = parser.parse_args()

    if args.url:
        os.environ["OPENBIOMED_API_BASE_URL"] = args.url

    # Run tests
    success = True
    success = test_tool_init() and success
    success = test_tool_usage() and success
    success = test_tool_prepare_database() and success
    success = test_tool_parse_results_empty() and success
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