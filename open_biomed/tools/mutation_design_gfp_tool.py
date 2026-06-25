"""GFP mutation design tool.

Thin specialization of :class:`MutationDesignBase` for the Green Fluorescent
Protein fluorescence oracle. All optimization logic lives in the base class;
only the download URLs, sequence length, and labels differ between AAV and GFP.

The oracle checkpoint is byte-identical to the official GGS repo
(github.com/kirjner/GGS) ``ckpt/GFP/.../cnn_oracle.ckpt`` and scores its own
training data at Spearman ~0.87 with the real BaseCNN forward.
"""

from open_biomed.tools.mutation_design_base import MutationDesignBase


class MutationDesignGFP(MutationDesignBase):
    """Design high-fluorescence GFP mutants through multi-round iterative
    optimization using the BaseCNN fitness oracle."""

    # Tsinghua-cloud URLs for the initial sequences, oracle checkpoint, and config.
    INITIAL_SEQUENCE_URL = "https://cloud.tsinghua.edu.cn/f/5e673c1db710466b828f/?dl=1"
    ORACLE_MODEL_URL = "https://cloud.tsinghua.edu.cn/f/f655f79d7bb04a98a0bb/?dl=1"
    ORACLE_CONFIG_URL = "https://cloud.tsinghua.edu.cn/f/8a894bb4b41f4074b9b0/?dl=1"

    SEQ_LEN = 237
    LABEL = "gfp"
    TASK_NAME = "GFP"


if __name__ == "__main__":
    # Test the tool
    tool = MutationDesignGFP()
    results, messages = tool.run(num_rounds=2)
    print(f"Results: {results}")
    print(f"Message: {messages[0]}")
