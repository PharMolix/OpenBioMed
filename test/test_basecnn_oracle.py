"""
Unit tests for the BaseCNN fitness oracle (vendored from GGS).

These tests cover the shared ``open_biomed/tools/basecnn_oracle.py`` module used by
the AAV/GFP mutation-design skills. The oracle was previously stubbed with a
``0.5 + random()`` placeholder; these tests pin the real forward so it cannot
silently regress to a non-functional scoring path.

Pure-logic tests (forward shape, integer encoding) need no checkpoint.
Integration tests against the trained checkpoint are skipped unless the cached
model/config files are present (they are validated to Spearman >= 0.4 on the
bundled initial-sequences file and >= 0.6 on the model's own training data).
"""

import os
import sys
import csv

import pytest
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_biomed.tools.basecnn_oracle import (
    BaseCNN,
    LengthMaxPool1D,
    encode_sequences,
    get_alphabet,
    load_oracle,
    score_sequences,
)

ALPHABET = "ARNDCQEGHILKMFPSTWYV"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AAV_CKPT = os.path.join(REPO_ROOT, "tmp", "aav_cache", "aav_oracle_model.pt")
AAV_CFG = os.path.join(REPO_ROOT, "tmp", "aav_cache", "aav_oracle_config.yaml")
AAV_INITIAL = os.path.join(REPO_ROOT, "tmp", "aav_cache", "aav_initial_sequences.txt")

have_aav_cache = os.path.exists(AAV_CKPT) and os.path.exists(AAV_CFG)


class TestBaseCNNForward:
    """Pure-logic tests for BaseCNN / LengthMaxPool1D (no checkpoint needed)."""

    def test_forward_output_shape(self):
        """BaseCNN forward on integer tokens (B, L) returns (B,)."""
        model = BaseCNN(n_tokens=20, kernel_size=5, input_size=256, linear=True)
        model.eval()
        tokens = torch.randint(0, 20, (8, 28))
        with torch.no_grad():
            out = model(tokens)
        assert out.shape == (8,)

    def test_forward_deterministic(self):
        """Same input -> identical output (eval mode, no dropout stochasticity)."""
        model = BaseCNN(n_tokens=20, kernel_size=5, input_size=256, linear=True)
        model.eval()
        tokens = torch.randint(0, 20, (4, 28))
        with torch.no_grad():
            out1 = model(tokens)
            out2 = model(tokens)
        assert torch.allclose(out1, out2)

    def test_forward_different_sequences_different_scores(self):
        """Distinct inputs should generally produce distinct outputs."""
        model = BaseCNN(n_tokens=20, kernel_size=5, input_size=256, linear=True)
        model.eval()
        tokens = torch.randint(0, 20, (16, 28))
        with torch.no_grad():
            out = model(tokens)
        # Not all outputs identical (random init still produces per-seq variance)
        assert not torch.allclose(out, out[0:1].expand_as(out))

    def test_length_max_pool_shape(self):
        """LengthMaxPool1D collapses the length axis, keeping the feature dim."""
        pool = LengthMaxPool1D(in_dim=256, out_dim=512, linear=True, activation="relu")
        x = torch.randn(4, 10, 256)
        out = pool(x)
        assert out.shape == (4, 512)

    def test_make_one_hot_false_accepts_onehot_input(self):
        """With make_one_hot=False the model accepts a pre-built (B, L, 20) tensor."""
        model = BaseCNN(n_tokens=20, kernel_size=5, input_size=256, make_one_hot=False)
        model.eval()
        onehot = torch.zeros(2, 28, 20)
        onehot[0, 0, 5] = 1.0
        onehot[1, 3, 10] = 1.0
        with torch.no_grad():
            out = model(onehot)
        assert out.shape == (2,)


class TestEncoding:
    """Tests for integer encoding under the training alphabet."""

    def test_encode_shape_and_values(self):
        """encode_sequences maps chars to their alphabet indices."""
        seqs = ["AR", "ND"]
        tokens = encode_sequences(seqs, ALPHABET)
        assert tokens.shape == (2, 2)
        assert tokens.dtype == torch.long
        # A->0, R->1 in ARNDCQEGHILKMFPSTWYV
        assert tokens.tolist() == [[0, 1], [2, 3]]

    def test_encode_alphabet_order_matters(self):
        """Reordering the alphabet changes the encoding (this is why it must match training)."""
        seqs = ["AR"]
        t1 = encode_sequences(seqs, "ARNDCQEGHILKMFPSTWYV")
        t2 = encode_sequences(seqs, "ACDEFGHIKLMNPQRSTVWY")
        assert not torch.equal(t1, t2)

    def test_get_alphabet_reads_config(self):
        """get_alphabet reads data.alphabet from a GGS config."""
        if not os.path.exists(AAV_CFG):
            pytest.skip("AAV config not cached")
        assert get_alphabet(AAV_CFG) == ALPHABET

    def test_encode_unknown_char_raises(self):
        """Chars outside the alphabet must raise (silent mis-encoding would destroy signal)."""
        with pytest.raises(KeyError):
            encode_sequences(["ARX"], ALPHABET)


@pytest.mark.skipif(not have_aav_cache, reason="AAV oracle checkpoint not cached")
class TestOracleIntegration:
    """Integration tests against the trained AAV checkpoint (skipped without cache)."""

    def test_load_oracle_returns_eval_basecnn(self):
        """load_oracle builds a BaseCNN with weights loaded and in eval mode."""
        model = load_oracle(AAV_CKPT, AAV_CFG, device="cpu")
        assert isinstance(model, BaseCNN)
        assert not model.training

    def test_score_sequences_deterministic(self):
        """Scoring is deterministic -> NOT the old 0.5+random() placeholder."""
        model = load_oracle(AAV_CKPT, AAV_CFG, device="cpu")
        seqs = ["ADEEIRATNPIATEMYGSVSTNLQLGNR"]
        s1 = score_sequences(model, seqs, ALPHABET)[0]
        s2 = score_sequences(model, seqs, ALPHABET)[0]
        assert s1 == pytest.approx(s2, abs=1e-6)

    def test_score_sequences_returns_floats(self):
        """score_sequences returns one float per input sequence."""
        model = load_oracle(AAV_CKPT, AAV_CFG, device="cpu")
        seqs = ["ADEEIRATNPIATEMYGSVSTNLQLGNR", "ADEEIRATNPIATEMYGSVSTNLQLGNA"]
        scores = score_sequences(model, seqs, ALPHABET)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_score_empty_list(self):
        """Empty input returns empty list (no crash)."""
        model = load_oracle(AAV_CKPT, AAV_CFG, device="cpu")
        assert score_sequences(model, [], ALPHABET) == []

    def test_oracle_correlates_with_ground_truth(self):
        """The oracle must rank the bundled initial sequences consistently with
        their measured fitness. Spearman >= 0.4 on the bundled slice (the model
        reaches ~0.95 on its own training data; the bundled slice is a harder,
        out-of-distribution query set, so the bar is lower but must stay positive).
        """
        try:
            from scipy.stats import spearmanr
        except ImportError:
            pytest.skip("scipy not available")
        if not os.path.exists(AAV_INITIAL):
            pytest.skip("AAV initial-sequences file not cached")

        seqs, gts = [], []
        with open(AAV_INITIAL, newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2 and len(row[0]) == 28 and set(row[0]).issubset(set(ALPHABET)):
                    seqs.append(row[0])
                    gts.append(float(row[1]))
        seqs, gts = seqs[:500], gts[:500]
        assert len(seqs) > 50

        model = load_oracle(AAV_CKPT, AAV_CFG, device="cpu")
        scores = score_sequences(model, seqs, ALPHABET)
        rho, _ = spearmanr(scores, gts)
        assert rho >= 0.4, f"oracle Spearman {rho:.3f} below 0.4 threshold — oracle may be broken"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
