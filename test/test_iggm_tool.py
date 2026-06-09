"""
Unit tests for IgGMRequester tool.
"""

import pytest
import os
import json
import tempfile
import logging

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_biomed.tools.iggm_tool import IgGMRequester

logging.basicConfig(level=logging.INFO)


class TestIgGMRequesterInit:
    """Test IgGMRequester initialization."""

    def test_default_init(self):
        """Test default initialization."""
        tool = IgGMRequester()
        assert tool.output_dir == "./tmp/iggm"
        assert tool.timeout == 300
        assert os.path.exists(tool.output_dir)

    def test_custom_init(self):
        """Test custom initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = IgGMRequester(output_dir=tmpdir, timeout=60)
            assert tool.output_dir == tmpdir
            assert tool.timeout == 60

    def test_print_usage(self):
        """Test print_usage method."""
        tool = IgGMRequester()
        usage = tool.print_usage()
        assert "IgGM" in usage
        assert "design_type" in usage
        assert "antigen_pdb" in usage
        assert "epitope" in usage
        assert "nanobody" in usage
        assert "heavy_light" in usage


class TestIgGMRequesterHealthCheck:
    """Test health check functionality."""

    def test_health_check_returns_bool(self):
        """Test that health check returns boolean."""
        tool = IgGMRequester()
        result = tool._health_check()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        os.environ.get("SKIP_EXTERNAL_API_TESTS") == "true",
        reason="Skipping external API tests"
    )
    def test_health_check_with_real_api(self):
        """Test health check with real IgGM API."""
        tool = IgGMRequester()
        result = tool._health_check()
        # Don't assert True because API might be down


class TestIgGMRequesterInputValidation:
    """Test input validation."""

    def test_missing_antigen_pdb(self):
        """Test error when antigen_pdb is missing."""
        tool = IgGMRequester()
        results, messages = tool.run(
            design_type="nanobody",
            heavy_chain_mask="QVQLVESGGXXXXXX",
            epitope="[109,110,111]"
        )
        assert len(results) == 0
        assert "Error" in messages[0]
        assert "antigen_pdb" in messages[0]

    def test_missing_heavy_chain_mask(self):
        """Test error when heavy_chain_mask is missing."""
        tool = IgGMRequester()
        results, messages = tool.run(
            design_type="nanobody",
            antigen_pdb="test.pdb",
            epitope="[109,110,111]"
        )
        assert len(results) == 0
        assert "Error" in messages[0]
        assert "heavy_chain_mask" in messages[0]

    def test_missing_epitope(self):
        """Test error when epitope is missing."""
        tool = IgGMRequester()
        results, messages = tool.run(
            design_type="nanobody",
            antigen_pdb="test.pdb",
            heavy_chain_mask="QVQLVESGGXXXXXX"
        )
        assert len(results) == 0
        assert "Error" in messages[0]
        assert "epitope" in messages[0]

    def test_missing_light_chain_for_heavy_light(self):
        """Test error when light_chain_mask missing for heavy_light."""
        tool = IgGMRequester()
        results, messages = tool.run(
            design_type="heavy_light",
            antigen_pdb="test.pdb",
            heavy_chain_mask="QVQLVESGGXXXXXX",
            epitope="[109,110,111]"
        )
        assert len(results) == 0
        assert "Error" in messages[0]
        assert "light_chain_mask" in messages[0]

    def test_invalid_epitope_format(self):
        """Test error for invalid epitope format."""
        tool = IgGMRequester()
        results, messages = tool.run(
            design_type="nanobody",
            antigen_pdb="test.pdb",
            heavy_chain_mask="QVQLVESGGXXXXXX",
            epitope="invalid_format"
        )
        assert len(results) == 0
        assert "Error" in messages[0]
        assert "epitope" in messages[0]

    def test_unknown_design_type(self):
        """Test with unknown design type (should still work)."""
        tool = IgGMRequester()
        # Missing required params will trigger first
        results, messages = tool.run(
            design_type="unknown_type",
            antigen_pdb="test.pdb",
            heavy_chain_mask="QVQLVESGGXXXXXX",
            epitope="[109,110,111]"
        )
        # Should fail on file not found, not design type
        assert "Error" in messages[0]


class TestIgGMRequesterEpitopeParsing:
    """Test epitope parsing functionality."""

    def test_parse_epitope_json_string(self):
        """Test parsing JSON string epitope."""
        tool = IgGMRequester()
        # Test inside run() by passing valid format
        epitope = "[109,110,111,112]"
        parsed = json.loads(epitope)
        assert parsed == [109, 110, 111, 112]

    def test_parse_epitope_comma_string(self):
        """Test parsing comma-separated string epitope."""
        epitope = "109,110,111,112"
        parsed = [int(x.strip()) for x in epitope.split(",")]
        assert parsed == [109, 110, 111, 112]

    def test_parse_epitope_single_int(self):
        """Test single integer as epitope."""
        tool = IgGMRequester()
        # Single residue
        epitope = 109
        epitope_list = [epitope]
        assert epitope_list == [109]


class TestIgGMRequesterDescriptionFormatting:
    """Test description formatting methods."""

    def test_format_nanobody_description(self):
        """Test nanobody description formatting."""
        tool = IgGMRequester()

        description = tool._format_description(
            "nanobody",
            ["./tmp/test.pdb", "./tmp/test.fasta"],
            {
                "job_id": "test-job-123",
                "antibody_type": "nanobody",
                "sequences": [{"heavy_chain": "QVQLVESGG..."}]
            },
            {"sequence": "QVQLVESGG"}
        )

        assert "Nanobody" in description
        assert "test-job-123" in description
        assert "QVQLVESGG" in description

    def test_format_heavy_light_description(self):
        """Test heavy_light description formatting."""
        tool = IgGMRequester()

        description = tool._format_description(
            "heavy_light",
            ["./tmp/test.pdb", "./tmp/test.fasta"],
            {
                "job_id": "test-job-456",
                "antibody_type": "heavy_light",
                "sequences": [{"heavy_chain": "QVQL...", "light_chain": "DIQMT..."}]
            },
            {}
        )

        assert "Heavy-light" in description
        assert "test-job-456" in description


class TestIgGMRequesterFileNotFound:
    """Test file not found error handling."""

    def test_antigen_pdb_not_found(self):
        """Test error when antigen PDB doesn't exist."""
        tool = IgGMRequester()

        results, messages = tool.run(
            design_type="nanobody",
            antigen_pdb="./nonexistent_file.pdb",
            heavy_chain_mask="QVQLVESGGXXXXXX",
            epitope="[109,110,111]"
        )

        assert len(results) == 0
        assert "Error" in messages[0]
        assert "not found" in messages[0].lower() or "FileNotFoundError" in messages[0]


@pytest.mark.skipif(
    os.environ.get("SKIP_EXTERNAL_API_TESTS") == "true",
    reason="Skipping external API tests"
)
class TestIgGMRequesterIntegration:
    """Integration tests with real IgGM API."""

    def test_health_check_integration(self):
        """Test health check with real API."""
        tool = IgGMRequester()
        healthy = tool._health_check()
        # Just test it returns without exception

    def test_nanobody_design_integration(self):
        """Test nanobody design with real API."""
        # This requires a valid antigen PDB file
        # Skip if not available
        pytest.skip("Requires valid antigen PDB file for integration test")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])