"""AAV mutation design tool.

Thin specialization of :class:`MutationDesignBase` for the AAV VP1 capsid
DNA-packaging fitness oracle. All optimization logic lives in the base class;
only the download URLs, sequence length, and labels differ between AAV and GFP.
"""

from open_biomed.tools.mutation_design_base import MutationDesignBase


class MutationDesignAAV(MutationDesignBase):
    """Design high-fitness AAV VP1 capsid protein mutants through multi-round
    iterative optimization using the BaseCNN fitness oracle."""

    # Tsinghua-cloud URLs for the initial sequences, oracle checkpoint, and config.
    # The checkpoint is byte-identical to the official GGS repo
    # (github.com/kirjner/GGS) ``ckpt/AAV/.../cnn_oracle.ckpt``.
    INITIAL_SEQUENCE_URL = "https://cloud.tsinghua.edu.cn/f/992109032d8049689a6d/?dl=1"
    ORACLE_MODEL_URL = "https://cloud.tsinghua.edu.cn/f/80bbc575ec3f4e63a0af/?dl=1"
    ORACLE_CONFIG_URL = "https://cloud.tsinghua.edu.cn/f/09ea0869b74b4d2ca53e/?dl=1"

    SEQ_LEN = 28
    LABEL = "aav"
    TASK_NAME = "AAV"


if __name__ == "__main__":
    # Test the tool
    tool = MutationDesignAAV()
    results, messages = tool.run(num_rounds=2)
    print(f"Results: {results}")
    print(f"Message: {messages[0]}")
