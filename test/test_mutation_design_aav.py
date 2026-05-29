"""
Unit tests for Mutation Design AAV Tool.

Tests the MutationDesignAAV tool functionality including:
- Tool initialization
- Mutant generation
- Sequence scoring
- Diversity calculation
- Full pipeline execution
"""

import pytest
import os
import tempfile
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_biomed.tools.mutation_design_aav_tool import MutationDesignAAV


class TestMutationDesignAAVInitialization:
    """Test MutationDesignAAV tool initialization."""

    def test_default_initialization(self):
        """Test default initialization with default parameters."""
        tool = MutationDesignAAV()
        assert tool.output_dir == "./tmp/mutation_design_aav"
        assert tool.cache_dir == "./tmp/aav_cache"
        assert tool._oracle_model is None
        assert tool._initial_sequences is None

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        custom_output = "/tmp/test_output"
        custom_cache = "/tmp/test_cache"
        tool = MutationDesignAAV(output_dir=custom_output, cache_dir=custom_cache)
        assert tool.output_dir == custom_output
        assert tool.cache_dir == custom_cache

    def test_print_usage(self):
        """Test print_usage method returns expected content."""
        tool = MutationDesignAAV()
        usage = tool.print_usage()
        assert "AAV Mutation Design" in usage
        assert "num_rounds" in usage
        assert "population_size" in usage
        assert "max_mutations" in usage


class TestMutantGeneration:
    """Test mutant generation functionality."""

    def test_generate_mutants_single_mutation(self):
        """Test generating mutants with single point mutation."""
        tool = MutationDesignAAV()
        sequence = "ACDEFGHIKLMNPQRSTVWY"  # 20 amino acids
        mutants = tool._generate_mutants([sequence], num_candidates=10, max_mutations=1)

        assert len(mutants) == 10
        for mutant in mutants:
            # Each mutant should have exactly 1 mutation
            hamming = tool._compute_hamming_distance(sequence, mutant)
            assert hamming == 1
            assert len(mutant) == len(sequence)

    def test_generate_mutants_multiple_mutations(self):
        """Test generating mutants with multiple mutations."""
        tool = MutationDesignAAV()
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        mutants = tool._generate_mutants([sequence], num_candidates=10, max_mutations=4)

        assert len(mutants) == 10
        for mutant in mutants:
            # Each mutant should have 1-4 mutations
            hamming = tool._compute_hamming_distance(sequence, mutant)
            assert 1 <= hamming <= 4

    def test_generate_mutants_from_population(self):
        """Test generating mutants from multiple sequences."""
        tool = MutationDesignAAV()
        sequences = ["ACDEFGHIKLMNPQRSTVWY", "VWQRSTPNMLKIHGFEDCA"]
        mutants = tool._generate_mutants(sequences, num_candidates=5, max_mutations=2)

        # Should generate 5 mutants per sequence = 10 total
        assert len(mutants) == 10


class TestDiversityCalculation:
    """Test diversity calculation functionality."""

    def test_compute_hamming_distance_identical(self):
        """Test Hamming distance for identical sequences."""
        tool = MutationDesignAAV()
        seq1 = "ACDEFGHIKLMNPQRSTVWY"
        seq2 = "ACDEFGHIKLMNPQRSTVWY"
        distance = tool._compute_hamming_distance(seq1, seq2)
        assert distance == 0

    def test_compute_hamming_distance_different(self):
        """Test Hamming distance for different sequences."""
        tool = MutationDesignAAV()
        seq1 = "ACDEFGHIKLMNPQRSTVWY"
        seq2 = "ACDEFGHIKLMNPQRSTVWX"  # Last char different
        distance = tool._compute_hamming_distance(seq1, seq2)
        assert distance == 1

    def test_compute_hamming_distance_multiple(self):
        """Test Hamming distance for multiple differences."""
        tool = MutationDesignAAV()
        seq1 = "AAAAA"
        seq2 = "BBBBB"
        distance = tool._compute_hamming_distance(seq1, seq2)
        assert distance == 5

    def test_compute_diversity_single_sequence(self):
        """Test diversity calculation with single sequence."""
        tool = MutationDesignAAV()
        diversity = tool._compute_diversity(["ACDEFGHIKLMNPQRSTVWY"])
        assert diversity == 0.0

    def test_compute_diversity_multiple_sequences(self):
        """Test diversity calculation with multiple sequences."""
        tool = MutationDesignAAV()
        sequences = ["AAAA", "BBBB", "CCCC"]
        diversity = tool._compute_diversity(sequences)
        # All sequences differ at all positions
        # Average pairwise distance = 4
        assert diversity == 4.0

    def test_compute_diversity_mixed_sequences(self):
        """Test diversity calculation with mixed sequences."""
        tool = MutationDesignAAV()
        sequences = ["AAAA", "AAAB", "AABB"]
        diversity = tool._compute_diversity(sequences)
        # Pairwise: (AAAA, AAAB) = 1, (AAAA, AABB) = 2, (AAAB, AABB) = 1
        # Average = (1 + 2 + 1) / 3 = 1.33
        assert diversity == 4 / 3


class TestSequenceScoring:
    """Test sequence scoring functionality."""

    def test_fallback_score_length_match(self):
        """Test fallback score for correct sequence length."""
        tool = MutationDesignAAV()
        sequence = "A" * 28  # Expected AAV segment length
        score = tool._fallback_score(sequence)
        assert 0.0 <= score <= 1.0

    def test_fallback_score_random_sequence(self):
        """Test fallback score for random sequence."""
        tool = MutationDesignAAV()
        sequence = "ACDEFGHIKLMNPQRSTVWYACDEF"
        score = tool._fallback_score(sequence)
        assert 0.0 <= score <= 1.0


class TestTopMutantSelection:
    """Test top mutant selection functionality."""

    def test_select_top_mutants_basic(self):
        """Test basic top mutant selection."""
        tool = MutationDesignAAV()
        sequences = ["AAA", "BBB", "CCC", "DDD"]
        scores = [0.9, 0.8, 0.7, 0.6]
        selected = tool._select_top_mutants(sequences, scores, population_size=2)
        assert len(selected) == 2
        assert selected[0] == "AAA"  # Highest score
        assert selected[1] == "BBB"

    def test_select_top_mutants_with_diversity(self):
        """Test top mutant selection with diversity consideration."""
        tool = MutationDesignAAV()
        sequences = ["AAAA", "AAAA", "BBBB", "CCCC"]
        scores = [0.9, 0.85, 0.8, 0.7]
        selected = tool._select_top_mutants(
            sequences, scores,
            population_size=3,
            diversity_weight=0.1
        )
        assert len(selected) <= 3
        # Should prefer diverse sequences over identical high-scoring ones

    def test_select_top_mutants_fill_population(self):
        """Test population filling when diversity filters too many."""
        tool = MutationDesignAAV()
        sequences = ["AAAA", "BBBB", "CCCC", "DDDD"]
        scores = [0.9, 0.8, 0.7, 0.6]
        selected = tool._select_top_mutants(
            sequences, scores,
            population_size=10,
            diversity_weight=0.1
        )
        # Should fill with all available sequences since we need 10
        assert len(selected) == 4


class TestFullPipeline:
    """Test full pipeline execution."""

    def test_run_basic(self):
        """Test basic pipeline run."""
        tool = MutationDesignAAV(output_dir=tempfile.mkdtemp())

        # Run with minimal rounds for testing
        # Note: serial_exec wraps output in nested lists: [[csv_path]], [[message]]
        results, messages = tool.run(num_rounds=1, population_size=10)

        assert len(results) == 1
        assert len(messages) == 1
        # results[0] is a list, results[0][0] is the csv path string
        csv_path = results[0][0]
        assert isinstance(csv_path, str)
        assert csv_path.endswith(".csv")
        assert "AAV mutation design completed" in messages[0][0]

        # Check CSV file exists
        assert os.path.exists(csv_path)

        # Clean up
        if os.path.exists(csv_path):
            os.remove(csv_path)

    def test_run_custom_parameters(self):
        """Test pipeline run with custom parameters."""
        tool = MutationDesignAAV(output_dir=tempfile.mkdtemp())

        results, messages = tool.run(
            num_rounds=2,
            population_size=20,
            max_mutations=3,
            diversity_weight=0.2
        )

        assert len(results) == 1
        assert len(messages) == 1

        # Check CSV file
        csv_path = results[0][0]
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            assert lines[0] == "sequence,fitness\n"
            # Should have 20 sequences (population_size)
            assert len(lines) <= 21  # Header + 20 sequences

        # Clean up
        if os.path.exists(csv_path):
            os.remove(csv_path)


class TestIntegration:
    """Integration tests for the MutationDesignAAV tool."""

    def test_tool_registry_integration(self):
        """Test that the tool can be accessed via TOOLS registry."""
        from open_biomed.tools.tool_registry import TOOLS

        # Check that mutation_design_aav is in available tools
        available = TOOLS.available_tools()
        assert "mutation_design_aav" in available

        # Check that the tool can be instantiated
        tool = TOOLS["mutation_design_aav"]
        assert isinstance(tool, MutationDesignAAV)

    def test_task_registry_integration(self):
        """Test that the task is registered in TASK_REGISTRY."""
        from open_biomed.tasks import TASK_REGISTRY

        assert "mutation_design_aav" in TASK_REGISTRY
        task_class = TASK_REGISTRY["mutation_design_aav"]
        assert task_class.print_usage() is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])