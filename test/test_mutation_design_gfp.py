"""
Unit tests for Mutation Design GFP Tool.

Mirrors test_mutation_design_aav.py for the GFP skill: tool initialization,
mutant generation, sequence scoring (real BaseCNN oracle), diversity, full
pipeline execution, and registry integration.
"""

import pytest
import os
import tempfile
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_biomed.tools.mutation_design_gfp_tool import MutationDesignGFP

# avGFP wild-type (237 aa) — used for oracle-scoring regression tests.
GFP_WT = (
    "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMK"
    "QHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQ"
    "KNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)


class TestMutationDesignGFPInitialization:
    """Test MutationDesignGFP tool initialization."""

    def test_default_initialization(self):
        """Test default initialization with default parameters."""
        tool = MutationDesignGFP()
        assert tool.output_dir == "./tmp/mutation_design_gfp"
        assert tool.cache_dir == "./tmp/gfp_cache"
        assert tool.SEQ_LEN == 237
        assert tool._oracle_model is None
        assert tool._initial_sequences is None

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        custom_output = "/tmp/test_output"
        custom_cache = "/tmp/test_cache"
        tool = MutationDesignGFP(output_dir=custom_output, cache_dir=custom_cache)
        assert tool.output_dir == custom_output
        assert tool.cache_dir == custom_cache

    def test_print_usage(self):
        """Test print_usage method returns expected content."""
        tool = MutationDesignGFP()
        usage = tool.print_usage()
        assert "GFP Mutation Design" in usage
        assert "num_rounds" in usage
        assert "population_size" in usage


class TestMutantGeneration:
    """Test mutant generation functionality."""

    def test_generate_mutants_single_mutation(self):
        tool = MutationDesignGFP()
        mutants = tool._generate_mutants([GFP_WT], num_candidates=10, max_mutations=1)
        assert len(mutants) == 10
        for mutant in mutants:
            assert tool._compute_hamming_distance(GFP_WT, mutant) == 1
            assert len(mutant) == len(GFP_WT)

    def test_generate_mutants_multiple_mutations(self):
        tool = MutationDesignGFP()
        mutants = tool._generate_mutants([GFP_WT], num_candidates=10, max_mutations=4)
        assert len(mutants) == 10
        for mutant in mutants:
            hamming = tool._compute_hamming_distance(GFP_WT, mutant)
            assert 1 <= hamming <= 4


class TestDiversityCalculation:
    """Test diversity calculation functionality."""

    def test_compute_hamming_distance_identical(self):
        tool = MutationDesignGFP()
        assert tool._compute_hamming_distance(GFP_WT, GFP_WT) == 0

    def test_compute_hamming_distance_different(self):
        tool = MutationDesignGFP()
        seq2 = GFP_WT[:-1] + ("A" if GFP_WT[-1] != "A" else "C")
        assert tool._compute_hamming_distance(GFP_WT, seq2) == 1

    def test_compute_diversity_single_sequence(self):
        tool = MutationDesignGFP()
        assert tool._compute_diversity([GFP_WT]) == 0.0

    def test_compute_diversity_multiple_sequences(self):
        tool = MutationDesignGFP()
        assert tool._compute_diversity(["AAAA", "BBBB", "CCCC"]) == 4.0


class TestSequenceScoring:
    """Test fallback scoring (oracle scoring is in TestOracleScoring)."""

    def test_fallback_score_length_match(self):
        """Fallback score for a 237-aa sequence (GFP length)."""
        tool = MutationDesignGFP()
        score = tool._fallback_score(GFP_WT)
        assert 0.0 <= score <= 1.0

    def test_fallback_score_short_sequence(self):
        """Fallback score for a non-GFP-length sequence."""
        tool = MutationDesignGFP()
        score = tool._fallback_score("ACDEFGHIKLMNPQRSTVWY")
        assert 0.0 <= score <= 1.0


class TestOracleScoring:
    """Regression guard for the real BaseCNN oracle scoring path.

    Skipped when the oracle checkpoint cannot be loaded (no cache / no network).
    """

    def test_score_sequence_deterministic(self):
        """Same sequence scored twice must give identical results (not random)."""
        tool = MutationDesignGFP()
        if tool._load_oracle_model() is None:
            pytest.skip("oracle checkpoint unavailable")
        s1 = tool._score_sequence(GFP_WT)
        s2 = tool._score_sequence(GFP_WT)
        assert s1 == pytest.approx(s2, abs=1e-6)

    def test_score_sequence_returns_float(self):
        tool = MutationDesignGFP()
        if tool._load_oracle_model() is None:
            pytest.skip("oracle checkpoint unavailable")
        score = tool._score_sequence(GFP_WT)
        assert isinstance(score, float)

    def test_different_sequences_different_scores(self):
        tool = MutationDesignGFP()
        if tool._load_oracle_model() is None:
            pytest.skip("oracle checkpoint unavailable")
        s1 = tool._score_sequence(GFP_WT)
        s2 = tool._score_sequence(GFP_WT[:-1] + ("A" if GFP_WT[-1] != "A" else "C"))
        assert s1 != pytest.approx(s2, abs=1e-4)


class TestTopMutantSelection:
    """Test top mutant selection functionality."""

    def test_select_top_mutants_basic(self):
        tool = MutationDesignGFP()
        sequences = ["AAA", "BBB", "CCC", "DDD"]
        scores = [0.9, 0.8, 0.7, 0.6]
        selected = tool._select_top_mutants(sequences, scores, population_size=2)
        assert len(selected) == 2
        assert selected[0] == "AAA"
        assert selected[1] == "BBB"


class TestFullPipeline:
    """Test full pipeline execution."""

    def test_run_basic(self):
        """Test basic pipeline run."""
        tool = MutationDesignGFP(output_dir=tempfile.mkdtemp())
        # serial_exec wraps output in nested lists: [[csv_path]], [[message]]
        results, messages = tool.run(num_rounds=1, population_size=10)

        assert len(results) == 1
        assert len(messages) == 1
        csv_path = results[0][0]
        assert isinstance(csv_path, str)
        assert csv_path.endswith(".csv")
        assert "GFP mutation design completed" in messages[0][0]
        assert os.path.exists(csv_path)

        if os.path.exists(csv_path):
            os.remove(csv_path)


class TestIntegration:
    """Integration tests for the MutationDesignGFP tool."""

    def test_tool_registry_integration(self):
        """Test that the tool can be accessed via TOOLS registry."""
        from open_biomed.tools.tool_registry import TOOLS
        available = TOOLS.available_tools()
        assert "mutation_design_gfp" in available
        tool = TOOLS["mutation_design_gfp"]
        assert isinstance(tool, MutationDesignGFP)

    def test_task_registry_integration(self):
        """Test that the task is registered in TASK_REGISTRY."""
        from open_biomed.tasks import TASK_REGISTRY
        assert "mutation_design_gfp" in TASK_REGISTRY
        task_class = TASK_REGISTRY["mutation_design_gfp"]
        assert task_class.print_usage() is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
