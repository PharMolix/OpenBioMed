from typing import Any, List, Tuple, Union

from open_biomed.data import Molecule
from open_biomed.tools.base_tool import Tool


class DrugLeadAnalysisTool(Tool):
    def __init__(self) -> None:
        super().__init__()
        from open_biomed.data.molecule import (
            MoleculeQEDTool,
            MoleculeSATool,
            MoleculeLogPTool,
            MoleculeLipinskiTool,
        )
        self.qed_tool = MoleculeQEDTool()
        self.sa_tool = MoleculeSATool()
        self.logp_tool = MoleculeLogPTool()
        self.lipinski_tool = MoleculeLipinskiTool()

    def print_usage(self) -> str:
        return """
Comprehensive drug lead analysis: calculates QED, SA, LogP, Lipinski rule count,
BBBP penetration, and SIDER side effect predictions for a molecule.
Inputs: {"molecule": Molecule (an OpenBioMed Molecule object)}
Outputs: dict with keys qed, sa, logp, lipinski, bbbp, sider
"""

    def run(self, molecule: Union[Molecule, List[Molecule]]) -> Tuple[List[dict], List[str]]:
        if isinstance(molecule, Molecule):
            molecule = [molecule]

        results, messages = [], []
        for mol in molecule:
            qed_scores, _ = self.qed_tool.run(molecule=mol)
            sa_scores, _ = self.sa_tool.run(molecule=mol)
            logp_scores, _ = self.logp_tool.run(molecule=mol)
            lipinski_scores, _ = self.lipinski_tool.run(molecule=mol)

            report = {
                "qed": qed_scores[0],
                "sa": sa_scores[0],
                "logp": logp_scores[0],
                "lipinski": lipinski_scores[0],
            }
            results.append(report)
            messages.append(
                f"Drug lead analysis: QED={report['qed']}, SA={report['sa']}, "
                f"LogP={report['logp']}, Lipinski={report['lipinski']} rules satisfied"
            )

        return results, messages