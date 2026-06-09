"""
Unit tests for TFoldRequester tool.
"""

import pytest
import os
import json
import tempfile
import logging

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_biomed.tools.tfold_tool import TFoldRequester

# Configure logging
logging.basicConfig(level=logging.INFO)


class TestTFoldRequesterInit:
    """Test TFoldRequester initialization."""

    def test_default_init(self):
        """Test default initialization."""
        tool = TFoldRequester()
        assert tool.output_dir == "./tmp/tfold"
        assert tool.timeout == 300
        assert os.path.exists(tool.output_dir)

    def test_custom_init(self):
        """Test custom initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = TFoldRequester(output_dir=tmpdir, timeout=60)
            assert tool.output_dir == tmpdir
            assert tool.timeout == 60

    def test_print_usage(self):
        """Test print_usage method."""
        tool = TFoldRequester()
        usage = tool.print_usage()
        assert "tFold Antibody Structure Prediction" in usage
        assert "prediction_type" in usage
        assert "antibody" in usage
        assert "nanobody" in usage
        assert "complex" in usage
        assert "epitope" in usage


class TestTFoldRequesterHealthCheck:
    """Test health check functionality."""

    def test_health_check_returns_bool(self):
        """Test that health check returns boolean."""
        tool = TFoldRequester()
        result = tool._health_check()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        os.environ.get("SKIP_EXTERNAL_API_TESTS") == "true",
        reason="Skipping external API tests"
    )
    def test_health_check_with_real_api(self):
        """Test health check with real tFold API."""
        tool = TFoldRequester()
        # This test may fail if API is unavailable
        result = tool._health_check()
        # We don't assert True because API might be down
        # Just check it doesn't raise exception


class TestTFoldRequesterValidation:
    """Test input validation."""

    def test_missing_heavy_chain_for_antibody(self):
        """Test error when heavy_chain is missing for antibody."""
        tool = TFoldRequester()
        results, messages = tool.run(
            prediction_type="antibody",
            light_chain="DIQMTQSPSSLSASVGDRVTITC"
        )
        assert len(results) == 0
        assert "Error" in messages[0]
        assert "heavy_chain" in messages[0]

    def test_missing_light_chain_for_antibody(self):
        """Test error when light_chain is missing for antibody."""
        tool = TFoldRequester()
        results, messages = tool.run(
            prediction_type="antibody",
            heavy_chain="EVQLVESGGGLVQPGGSLRLSCAAS"
        )
        assert len(results) == 0
        assert "Error" in messages[0]
        assert "light_chain" in messages[0]

    def test_missing_heavy_chain_for_nanobody(self):
        """Test error when heavy_chain is missing for nanobody."""
        tool = TFoldRequester()
        results, messages = tool.run(
            prediction_type="nanobody"
        )
        assert len(results) == 0
        assert "Error" in messages[0]
        assert "heavy_chain" in messages[0]

    def test_missing_antigen_for_complex(self):
        """Test error when antigen is missing for complex."""
        tool = TFoldRequester()
        results, messages = tool.run(
            prediction_type="complex",
            heavy_chain="EVQLVESGGGLVQPGGSLRLSCAAS",
            light_chain="DIQMTQSPSSLSASVGDRVTITC"
        )
        assert len(results) == 0
        assert "Error" in messages[0]
        assert "antigen" in messages[0]

    def test_missing_pdb_file_for_epitope(self):
        """Test error when pdb_file is missing for epitope."""
        tool = TFoldRequester()
        results, messages = tool.run(
            prediction_type="epitope"
        )
        assert len(results) == 0
        assert "Error" in messages[0]
        assert "pdb_file" in messages[0]

    def test_unknown_prediction_type(self):
        """Test error for unknown prediction_type."""
        tool = TFoldRequester()
        results, messages = tool.run(
            prediction_type="unknown_type"
        )
        assert len(results) == 0
        assert "Error" in messages[0]
        assert "Unknown prediction_type" in messages[0]


class TestTFoldRequesterConfidenceExtraction:
    """Test confidence score extraction from PDB."""

    def test_extract_confidence_scores(self):
        """Test extracting confidence scores from PDB content."""
        tool = TFoldRequester()

        # Create a mock PDB file with confidence scores
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_file = os.path.join(tmpdir, "test.pdb")
            with open(pdb_file, 'w') as f:
                f.write("REMARK 250 Predicted lDDT-Ca score: 0.9482\n")
                f.write("REMARK 250 Predicted pTM score: 0.7861\n")
                f.write("REMARK 250 Predicted ipTM score: 0.8023\n")
                f.write("ATOM      1  N   MET A   1      10.0   10.0   10.0\n")

            metadata = tool._extract_confidence_scores(pdb_file)

            assert metadata.get('lddt_ca') == 0.9482
            assert metadata.get('ptm') == 0.7861
            assert metadata.get('iptm') == 0.8023

    def test_extract_confidence_scores_missing(self):
        """Test extracting confidence scores when they are missing."""
        tool = TFoldRequester()

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_file = os.path.join(tmpdir, "test_no_scores.pdb")
            with open(pdb_file, 'w') as f:
                f.write("ATOM      1  N   MET A   1      10.0   10.0   10.0\n")

            metadata = tool._extract_confidence_scores(pdb_file)
            assert len(metadata) == 0


class TestTFoldRequesterDescriptionFormatting:
    """Test description formatting methods."""

    def test_format_antibody_description(self):
        """Test antibody description formatting."""
        tool = TFoldRequester()

        description = tool._format_antibody_description(
            "antibody",
            "./tmp/test.pdb",
            {"lddt_ca": 0.95, "ptm": 0.8},
            {"sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYYMAWVRQAPGKGLEWVSAISSSGGSTYYADSVKGRLTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYKHNWFGTEVTVELTK"}
        )

        assert "Antibody structure prediction completed" in description
        assert "./tmp/test.pdb" in description
        assert "lDDT-Ca score: 0.95" in description
        assert "pTM score: 0.8" in description
        assert "Sequence length:" in description

    def test_format_nanobody_description(self):
        """Test nanobody description formatting."""
        tool = TFoldRequester()

        description = tool._format_antibody_description(
            "nanobody",
            "./tmp/nanobody.pdb",
            {"lddt_ca": 0.85},
            {}
        )

        assert "Nanobody structure prediction completed" in description
        assert "./tmp/nanobody.pdb" in description
        assert "lDDT-Ca score: 0.85" in description

    def test_format_complex_description(self):
        """Test complex description formatting."""
        tool = TFoldRequester()

        description = tool._format_complex_description(
            "./tmp/complex.pdb",
            {"lddt_ca": 0.92, "iptm": 0.80},
            {"sequence": "ABCDEF"},
            "A"
        )

        assert "Antigen-antibody complex" in description
        assert "./tmp/complex.pdb" in description
        assert "Antigen chain ID: A" in description
        assert "ipTM score: 0.80" in description

    def test_format_epitope_description(self):
        """Test epitope description formatting."""
        tool = TFoldRequester()

        result = {
            "epitope_count": 65,
            "epitope_residues": [
                [27, "SER", "A"],
                [28, "ALA", "A"],
                [29, "GLY", "A"]
            ]
        }

        description = tool._format_epitope_description(
            result, "A", 5.0
        )

        assert "Epitope determination completed" in description
        assert "Antigen chain ID: A" in description
        assert "Distance threshold: 5.0" in description
        assert "Epitope residue count: 65" in description


class TestTFoldRequesterEpitopeFileNotFound:
    """Test epitope prediction with missing file."""

    def test_epitope_file_not_found(self):
        """Test epitope prediction when PDB file doesn't exist."""
        tool = TFoldRequester()

        # Call _predict_epitope directly to test FileNotFoundError
        with pytest.raises(FileNotFoundError):
            tool._predict_epitope(
                pdb_file="./nonexistent_file.pdb",
                antigen_id="A",
                distance_threshold=5.0
            )


@pytest.mark.skipif(
    os.environ.get("SKIP_EXTERNAL_API_TESTS") == "true",
    reason="Skipping external API tests"
)
class TestTFoldRequesterIntegration:
    """Integration tests with real tFold API."""

    def test_antibody_prediction_integration(self):
        """Test antibody prediction with real API."""
        tool = TFoldRequester()

        results, messages = tool.run(
            prediction_type="antibody",
            heavy_chain="EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYYMAWVRQAPGKGLEWVSAISSSGGSTYYADSVKGRLTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYKHNWFGTEVTVELTK",
            light_chain="DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPPTFGQGTKVEIK",
            output_name="test_antibody_integration"
        )

        # Check that we got results
        assert len(results) > 0
        assert len(messages) > 0

        # Check that PDB file was created
        pdb_file = results[0]
        assert pdb_file.endswith(".pdb")
        assert os.path.exists(pdb_file)

    def test_nanobody_prediction_integration(self):
        """Test nanobody prediction with real API."""
        tool = TFoldRequester()

        results, messages = tool.run(
            prediction_type="nanobody",
            heavy_chain="MSIQEIQKEIAQIQAVIAGIQKYIYTMSIEEIQKQIAAIQCQIAAIQKQIYAMSIEEIQKQIAAIQEQILAIYKQIMAMVT",
            output_name="test_nanobody_integration"
        )

        assert len(results) > 0
        assert len(messages) > 0
        assert os.path.exists(results[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])