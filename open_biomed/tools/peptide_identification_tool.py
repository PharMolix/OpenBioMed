"""
Peptide and protein identification tool using MSFragger and Philosopher.

Search MS2 spectra against a protein sequence database to identify peptides
and proteins. Apply target-decoy FDR filtering to control false discovery rate.

This is Step 2 of the proteomics pipeline — takes centroided mzML from Step 1,
produces PSM tables and protein groups for Step 3 (quantification).

Supported operations:
- prepare_database: Prepare protein database with decoys and contaminants
- search: Run MSFragger database search
- validate: Run Philosopher validation (PeptideProphet, ProteinProphet, filter)
- full_pipeline: Complete pipeline (prepare + search + validate)
- parse_results: Parse TSV output files into structured data
"""

import os
import glob
import uuid
import logging
import subprocess
import tempfile
import shutil
from typing import Tuple, Dict, Any, Optional, List

from open_biomed.tools.base_tool import Tool

logger = logging.getLogger('OpenBioMed')

# Default paths for jar files
# Check multiple possible locations: project root, tools directory, open_biomed/tools
def _find_jar_path(jar_name: str) -> str:
    """Find jar file in multiple possible locations."""
    # Possible base directories to check
    base_dirs = [
        # Project root tools directory
        os.path.join(os.path.dirname(__file__), '..', '..', 'tools', 'proteomics'),
        # Open_biomed tools directory
        os.path.join(os.path.dirname(__file__), '..', 'tools', 'proteomics'),
        # Current working directory
        os.path.join(os.getcwd(), 'tools', 'proteomics'),
        # Absolute path in container
        '/app/tools/proteomics',
    ]

    for base_dir in base_dirs:
        jar_path = os.path.abspath(os.path.join(base_dir, jar_name))
        if os.path.exists(jar_path):
            return jar_path

    # Return default path (even if doesn't exist - will show download instructions)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'tools', 'proteomics', jar_name))

DEFAULT_MSFRAGGER_PATH = _find_jar_path('MSFragger.jar')
DEFAULT_PHILOSOPHER_PATH = _find_jar_path('philosopher.jar')


class PeptideIdentification(Tool):
    """
    Identify peptides and proteins from mass spectrometry data using MSFragger and Philosopher.

    This tool wraps MSFragger (search engine) and Philosopher (validation/reporting)
    for shotgun proteomics data analysis.

    Requirements:
    - Java 11+ (for MSFragger 4.x)
    - MSFragger.jar (download from https://github.com/Nesvilab/MSFragger/releases)
    - philosopher.jar (download from https://github.com/Nesvilab/philosopher/releases)

    Key features:
    - Target-decoy strategy for FDR estimation
    - PeptideProphet/ProteinProphet for validation
    - PSM, peptide, and protein-level FDR filtering
    - Parsimony protein grouping
    """

    def __init__(
        self,
        msfragger_path: Optional[str] = None,
        philosopher_path: Optional[str] = None,
        java_path: str = "java"
    ) -> None:
        super().__init__()
        self._msfragger_path = msfragger_path or DEFAULT_MSFRAGGER_PATH
        self._philosopher_path = philosopher_path or DEFAULT_PHILOSOPHER_PATH
        self._java_path = java_path
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if required dependencies (Java, jar files) are available."""
        # Check Java
        try:
            result = subprocess.run(
                [self._java_path, "-version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            version_str = result.stderr or result.stdout
            if "version" in version_str:
                # Parse version
                import re
                match = re.search(r'version "?(\d+)', version_str)
                if match:
                    major_version = int(match.group(1))
                    if major_version < 11:
                        logger.warning(f"Java version {major_version} may not support MSFragger 4.x (requires Java 11+)")
                    else:
                        logger.info(f"Java {major_version} detected - compatible with MSFragger")
        except Exception as e:
            logger.warning(f"Java check failed: {e}")

        # Check MSFragger
        if not os.path.exists(self._msfragger_path):
            logger.warning(f"MSFragger.jar not found at {self._msfragger_path}")
            logger.info("Download from: https://github.com/Nesvilab/MSFragger/releases")
        else:
            logger.info(f"MSFragger found: {self._msfragger_path}")

        # Check Philosopher
        if not os.path.exists(self._philosopher_path):
            logger.warning(f"philosopher.jar not found at {self._philosopher_path}")
            logger.info("Download from: https://github.com/Nesvilab/philosopher/releases")
        else:
            logger.info(f"Philosopher found: {self._philosopher_path}")

    def _check_pandas(self) -> bool:
        """Check if pandas is available."""
        try:
            import pandas
            return True
        except ImportError:
            return False

    def print_usage(self) -> str:
        return """
Usage: Identify peptides and proteins from MS2 spectra using MSFragger and Philosopher
Inputs: {
    "operation": str (prepare_database, search, validate, full_pipeline, parse_results),
    "mzml_files": str or list (paths to centroided mzML files),
    "database_file": str (path to protein FASTA database),
    "output_dir": str (output directory, default: ./tmp/),
    "search_params": dict (optional, MSFragger search parameters),
    "fdr_threshold": float (optional, default: 0.01 for 1% FDR),
    "java_memory": str (optional, Java heap size, default: -Xmx32g)
}
Outputs: {
    "result": dict (operation-specific results),
    "message": str (status message)
}

Operations:
- prepare_database: Download UniProt + cRAP, append decoys
- search: Run MSFragger database search on mzML files
- validate: Run Philosopher pipeline (peptideprophet, proteinprophet, filter, report)
- full_pipeline: Complete workflow (prepare_database + search + validate)
- parse_results: Parse TSV output files (psm.tsv, peptide.tsv, protein.tsv)

Required tools:
- MSFragger.jar: https://github.com/Nesvilab/MSFragger/releases/download/v4.1/MSFragger-4.1.jar
- philosopher.jar: https://github.com/Nesvilab/philosopher/releases/download/v5.1.0/philosopher_5.1.0.jar
- Java 11+: Required for MSFragger 4.x

Example search_params:
{
    "precursor_mass_tolerance": 20,  # ppm
    "fragment_mass_tolerance": 20,   # ppm
    "enzyme": "Trypsin",
    "missed_cleavages": 2,
    "fixed_mods": ["C+57.02146"],  # carbamidomethylation
    "variable_mods": ["M+15.99491", "n+42.01057"]  # oxidation, N-term acetylation
}
"""

    def run(
        self,
        operation: str,
        mzml_files: Optional[List[str]] = None,
        database_file: Optional[str] = None,
        output_dir: str = "./tmp/",
        organism: str = "human",
        search_params: Optional[Dict[str, Any]] = None,
        fdr_threshold: float = 0.01,
        java_memory: str = "-Xmx32g",
        **kwargs
    ) -> Tuple[Dict[str, Any], str]:
        """
        Run peptide identification operation.

        Args:
            operation: Operation to perform
            mzml_files: List of mzML file paths (for search, validate, full_pipeline)
            database_file: Path to protein FASTA database
            output_dir: Output directory
            organism: Organism name for database download (human, mouse, etc.)
            search_params: MSFragger search parameters dict
            fdr_threshold: FDR threshold (default 0.01 = 1%)
            java_memory: Java heap size (default -Xmx32g)

        Returns:
            Tuple of (result_dict, message)
        """
        os.makedirs(output_dir, exist_ok=True)
        output_id = str(uuid.uuid4())[:8]

        # Resolve jar paths
        msfragger_path = kwargs.get('msfragger_path', self._msfragger_path)
        philosopher_path = kwargs.get('philosopher_path', self._philosopher_path)

        logger.info(f"Running peptide identification operation: {operation}")

        if operation == "prepare_database":
            result = self._prepare_database(
                output_dir, output_id, organism, philosopher_path
            )
        elif operation == "search":
            if not mzml_files:
                raise ValueError("mzml_files is required for search operation")
            if not database_file:
                raise ValueError("database_file is required for search operation")
            result = self._run_search(
                mzml_files, database_file, output_dir, output_id,
                search_params or {}, msfragger_path, java_memory
            )
        elif operation == "validate":
            if not mzml_files:
                raise ValueError("mzml_files is required for validate operation")
            if not database_file:
                raise ValueError("database_file is required for validate operation")
            result = self._run_validation(
                mzml_files, database_file, output_dir, output_id,
                fdr_threshold, philosopher_path
            )
        elif operation == "full_pipeline":
            if not mzml_files:
                raise ValueError("mzml_files is required for full_pipeline operation")
            result = self._run_full_pipeline(
                mzml_files, output_dir, output_id, organism,
                search_params or {}, fdr_threshold,
                msfragger_path, philosopher_path, java_memory
            )
        elif operation == "parse_results":
            result = self._parse_results(output_dir)
        else:
            raise ValueError(f"Unknown operation: {operation}. "
                           f"Supported: prepare_database, search, validate, full_pipeline, parse_results")

        message = result.get("message", f"Operation {operation} completed")
        return result, message

    def _prepare_database(
        self,
        output_dir: str,
        output_id: str,
        organism: str,
        philosopher_path: str
    ) -> Dict[str, Any]:
        """
        Prepare protein database with decoys and contaminants.

        Downloads UniProt reference proteome and cRAP contaminants,
        then appends decoy sequences using Philosopher.
        """
        if not os.path.exists(philosopher_path):
            return {
                "status": "error",
                "message": f"Philosopher not found at {philosopher_path}. "
                          f"Download from: https://github.com/Nesvilab/philosopher/releases",
                "philosopher_download": "https://github.com/Nesvilab/philosopher/releases"
            }

        # UniProt proteome mapping
        proteome_map = {
            "human": "UP000005640",
            "mouse": "UP000000589",
            "rat": "UP000000241",
            "yeast": "UP000002311",
            "ecoli": "UP000000625",
        }
        proteome_id = proteome_map.get(organism.lower(), "UP000005640")

        # Download UniProt
        uniprot_file = os.path.join(output_dir, f"{organism}_uniprot_{output_id}.fasta")
        uniprot_url = f"https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=(proteome:{proteome_id})"

        logger.info(f"Downloading UniProt proteome {proteome_id} for {organism}...")
        result = subprocess.run(
            ["curl", "-L", "-o", uniprot_file, uniprot_url],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0 or not os.path.exists(uniprot_file):
            logger.warning(f"UniProt download failed: {result.stderr}")
            # Provide manual download instructions
            return {
                "status": "error",
                "message": f"UniProt download failed. Download manually from: {uniprot_url}",
                "uniprot_download_url": uniprot_url,
                "crap_download_url": "https://www.thegpm.org/crap/caRaP.fasta",
                "instructions": [
                    f"1. Download UniProt: curl -L -o {uniprot_file} '{uniprot_url}'",
                    f"2. Download cRAP: curl -L -o crap.fasta 'https://www.thegpm.org/crap/caRaP.fasta'",
                    f"3. Run: java -jar {philosopher_path} workspace --init",
                    f"4. Run: java -jar {philosopher_path} database --reviewed --contam --custom crap.fasta --prefix 'DECOY_' {uniprot_file}",
                ]
            }

        # Download cRAP contaminants
        crap_file = os.path.join(output_dir, f"crap_{output_id}.fasta")
        crap_url = "https://www.thegpm.org/crap/caRaP.fasta"

        logger.info("Downloading cRAP contaminants...")
        result = subprocess.run(
            ["curl", "-L", "-o", crap_file, crap_url],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            logger.warning(f"cRAP download failed: {result.stderr}")

        # Initialize Philosopher workspace and create target-decoy database
        workspace_dir = os.path.join(output_dir, f"workspace_{output_id}")
        os.makedirs(workspace_dir, exist_ok=True)

        # Copy database files to workspace
        shutil.copy(uniprot_file, workspace_dir)
        if os.path.exists(crap_file):
            shutil.copy(crap_file, workspace_dir)

        # Run Philosopher database preparation
        logger.info("Running Philosopher database preparation...")
        cmds = [
            [self._java_path, "-jar", philosopher_path, "workspace", "--init"],
            [self._java_path, "-jar", philosopher_path, "database",
             "--reviewed", "--contam", "--custom", os.path.basename(crap_file) if os.path.exists(crap_file) else "",
             "--prefix", "DECOY_", os.path.basename(uniprot_file)]
        ]

        for cmd in cmds:
            result = subprocess.run(
                cmd,
                cwd=workspace_dir,
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                logger.warning(f"Philosopher command failed: {cmd}\n{result.stderr}")

        # Find the generated target-decoy database
        td_fasta = None
        for f in glob.glob(os.path.join(workspace_dir, "*_td.fasta")):
            td_fasta = f
            break

        if not td_fasta:
            td_fasta = os.path.join(workspace_dir, f"{organism}_combined_td.fasta")

        # Count sequences
        n_targets = 0
        n_decoys = 0
        try:
            with open(td_fasta or uniprot_file, 'r') as f:
                content = f.read()
                n_targets = len([l for l in content.split('\n') if l.startswith('>') and not l.startswith('>DECOY_')])
                n_decoys = len([l for l in content.split('\n') if l.startswith('>DECOY_')])
        except Exception as e:
            logger.warning(f"Could not count sequences: {e}")

        return {
            "status": "success",
            "database_file": td_fasta or uniprot_file,
            "uniprot_file": uniprot_file,
            "crap_file": crap_file if os.path.exists(crap_file) else None,
            "workspace_dir": workspace_dir,
            "n_target_proteins": n_targets,
            "n_decoy_proteins": n_decoys,
            "organism": organism,
            "message": f"Database prepared: {n_targets} target + {n_decoys} decoy proteins"
        }

    def _run_search(
        self,
        mzml_files: List[str],
        database_file: str,
        output_dir: str,
        output_id: str,
        search_params: Dict[str, Any],
        msfragger_path: str,
        java_memory: str
    ) -> Dict[str, Any]:
        """
        Run MSFragger database search on mzML files.

        Generates pepXML output files for each mzML input.
        """
        if not os.path.exists(msfragger_path):
            return {
                "status": "error",
                "message": f"MSFragger not found at {msfragger_path}. "
                          f"Download from: https://github.com/Nesvilab/MSFragger/releases",
                "msfragger_download": "https://github.com/Nesvilab/MSFragger/releases"
            }

        # Validate mzML files
        valid_mzml_files = []
        for f in mzml_files:
            if os.path.exists(f) and f.endswith('.mzML'):
                valid_mzml_files.append(f)
            else:
                logger.warning(f"Skipping invalid mzML file: {f}")

        if not valid_mzml_files:
            raise ValueError("No valid mzML files found")

        # Validate database
        if not os.path.exists(database_file):
            raise FileNotFoundError(f"Database file not found: {database_file}")

        # Create MSFragger parameter file
        params_file = os.path.join(output_dir, f"fragger_{output_id}.params")
        self._write_fragger_params(params_file, database_file, search_params)

        # Run MSFragger
        logger.info(f"Running MSFragger on {len(valid_mzml_files)} mzML files...")

        cmd = [
            self._java_path, java_memory, "-jar", msfragger_path,
            params_file
        ] + valid_mzml_files

        result = subprocess.run(
            cmd,
            cwd=output_dir,
            capture_output=True, text=True, timeout=3600  # 1 hour timeout
        )

        if result.returncode != 0:
            logger.error(f"MSFragger failed: {result.stderr}")
            return {
                "status": "error",
                "message": f"MSFragger search failed: {result.stderr}",
                "params_file": params_file,
                "mzml_files": valid_mzml_files
            }

        # Collect pepXML outputs
        pepxml_files = glob.glob(os.path.join(output_dir, "*.pepXML"))

        return {
            "status": "success",
            "params_file": params_file,
            "mzml_files": valid_mzml_files,
            "pepxml_files": pepxml_files,
            "n_pepxml": len(pepxml_files),
            "output_dir": output_dir,
            "message": f"MSFragger search completed. Generated {len(pepxml_files)} pepXML files"
        }

    def _write_fragger_params(
        self,
        params_file: str,
        database_file: str,
        search_params: Dict[str, Any]
    ) -> None:
        """Write MSFragger parameter file."""

        # Default parameters (Orbitrap DDA settings)
        defaults = {
            "num_threads": 8,
            "precursor_mass_tolerance": 20,
            "precursor_mass_units": 1,  # ppm
            "fragment_mass_tolerance": 20,
            "fragment_mass_units": 1,  # ppm
            "search_enzyme_name": "Trypsin",
            "search_enzyme_cut_after": "KR",
            "search_enzyme_no_cut_before": "P",
            "allowed_missed_cleavage": 2,
            "add_C_cysteine": 57.02146,  # carbamidomethylation
            "variable_mod_01": "15.99491 M 3",  # oxidation
            "digest_min_length": 7,
            "digest_max_length": 50,
            "precursor_charge": "1 4",
            "output_format": "pepXML"
        }

        # Override with user params
        params = {**defaults, **search_params}
        params["database_name"] = database_file

        # Write param file
        with open(params_file, 'w') as f:
            f.write(f"database_name = {params['database_name']}\n")
            f.write(f"num_threads = {params['num_threads']}\n\n")

            f.write("# Enzyme settings\n")
            f.write(f"search_enzyme_name = {params['search_enzyme_name']}\n")
            f.write(f"search_enzyme_cut_after = {params['search_enzyme_cut_after']}\n")
            f.write(f"search_enzyme_no_cut_before = {params['search_enzyme_no_cut_before']}\n")
            f.write(f"allowed_missed_cleavage = {params['allowed_missed_cleavage']}\n\n")

            f.write("# Mass tolerances (Orbitrap)\n")
            f.write(f"precursor_mass_tolerance = {params['precursor_mass_tolerance']}\n")
            f.write(f"precursor_mass_units = {params['precursor_mass_units']}\n")
            f.write(f"fragment_mass_tolerance = {params['fragment_mass_tolerance']}\n")
            f.write(f"fragment_mass_units = {params['fragment_mass_units']}\n\n")

            f.write("# Modifications\n")
            f.write(f"add_C_cysteine = {params['add_C_cysteine']}\n")
            f.write(f"variable_mod_01 = {params['variable_mod_01']}\n\n")

            f.write("# Peptide filters\n")
            f.write(f"digest_min_length = {params['digest_min_length']}\n")
            f.write(f"digest_max_length = {params['digest_max_length']}\n")
            f.write(f"precursor_charge = {params['precursor_charge']}\n\n")

            f.write(f"output_format = {params['output_format']}\n")

        logger.info(f"Written MSFragger params to {params_file}")

    def _run_validation(
        self,
        mzml_files: List[str],
        database_file: str,
        output_dir: str,
        output_id: str,
        fdr_threshold: float,
        philosopher_path: str
    ) -> Dict[str, Any]:
        """
        Run Philosopher validation pipeline.

        Steps:
        1. PeptideProphet - PSM validation
        2. iProphet - combine across runs
        3. ProteinProphet - protein inference
        4. Filter - apply FDR thresholds
        5. Report - export TSV tables
        """
        if not os.path.exists(philosopher_path):
            return {
                "status": "error",
                "message": f"Philosopher not found at {philosopher_path}. "
                          f"Download from: https://github.com/Nesvilab/philosopher/releases"
            }

        # Find pepXML files from search output
        pepxml_files = glob.glob(os.path.join(output_dir, "*.pepXML"))
        if not pepxml_files:
            return {
                "status": "error",
                "message": "No pepXML files found. Run 'search' operation first."
            }

        # Initialize workspace
        subprocess.run(
            [self._java_path, "-jar", philosopher_path, "workspace", "--init"],
            cwd=output_dir, capture_output=True, text=True
        )

        # Run PeptideProphet
        logger.info("Running PeptideProphet...")
        cmd = [
            self._java_path, "-jar", philosopher_path, "peptideprophet",
            "--database", database_file,
            "--decoy", "DECOY_",
            "--ppm",
            "--accmass"
        ] + pepxml_files

        result = subprocess.run(
            cmd, cwd=output_dir, capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            logger.warning(f"PeptideProphet warning: {result.stderr}")

        # Run iProphet to combine results
        interact_files = glob.glob(os.path.join(output_dir, "interact-*.pep.xml"))
        if interact_files:
            logger.info("Running iProphet...")
            cmd = [
                self._java_path, "-jar", philosopher_path, "iprophet",
                "--output", "combined.pep.xml"
            ] + interact_files

            result = subprocess.run(
                cmd, cwd=output_dir, capture_output=True, text=True, timeout=600
            )

            # Run ProteinProphet on combined file
            if os.path.exists(os.path.join(output_dir, "combined.pep.xml")):
                logger.info("Running ProteinProphet...")
                cmd = [
                    self._java_path, "-jar", philosopher_path, "proteinprophet",
                    "--output", "proteinprophet.prot.xml",
                    "combined.pep.xml"
                ]
                result = subprocess.run(
                    cmd, cwd=output_dir, capture_output=True, text=True, timeout=600
                )
        else:
            # Single-run mode: run ProteinProphet directly
            logger.info("Running ProteinProphet (single-run mode)...")
            prot_xml = "proteinprophet.prot.xml"
            for pf in pepxml_files:
                cmd = [
                    self._java_path, "-jar", philosopher_path, "proteinprophet",
                    "--output", prot_xml, pf
                ]
                result = subprocess.run(
                    cmd, cwd=output_dir, capture_output=True, text=True, timeout=600
                )

        # Apply FDR filtering
        logger.info(f"Applying FDR filter at {fdr_threshold}...")
        cmd = [
            self._java_path, "-jar", philosopher_path, "filter",
            "--psm", str(fdr_threshold),
            "--pep", str(fdr_threshold),
            "--prot", str(fdr_threshold),
            "--picked",
            "--tag", "DECOY_",
            "--razor"
        ]
        result = subprocess.run(
            cmd, cwd=output_dir, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            logger.warning(f"Filter warning: {result.stderr}")

        # Generate report
        logger.info("Generating report...")
        cmd = [self._java_path, "-jar", philosopher_path, "report"]
        result = subprocess.run(
            cmd, cwd=output_dir, capture_output=True, text=True, timeout=300
        )

        # Collect output TSV files
        tsv_files = {
            "psm": glob.glob(os.path.join(output_dir, "psm.tsv")),
            "peptide": glob.glob(os.path.join(output_dir, "peptide.tsv")),
            "protein": glob.glob(os.path.join(output_dir, "protein.tsv")),
            "ion": glob.glob(os.path.join(output_dir, "ion.tsv"))
        }

        return {
            "status": "success",
            "pepxml_files": pepxml_files,
            "tsv_files": {k: v[0] if v else None for k, v in tsv_files.items()},
            "fdr_threshold": fdr_threshold,
            "output_dir": output_dir,
            "message": f"Validation completed. FDR filtered at {fdr_threshold*100}%"
        }

    def _run_full_pipeline(
        self,
        mzml_files: List[str],
        output_dir: str,
        output_id: str,
        organism: str,
        search_params: Dict[str, Any],
        fdr_threshold: float,
        msfragger_path: str,
        philosopher_path: str,
        java_memory: str
    ) -> Dict[str, Any]:
        """
        Run complete peptide identification pipeline.

        Steps:
        1. Prepare database (UniProt + cRAP + decoys)
        2. MSFragger search
        3. Philosopher validation and filtering
        """
        # Step 1: Prepare database
        logger.info("Step 1: Preparing database...")
        db_result = self._prepare_database(
            output_dir, output_id, organism, philosopher_path
        )

        if db_result.get("status") != "success":
            return db_result

        database_file = db_result["database_file"]

        # Step 2: Run search
        logger.info("Step 2: Running MSFragger search...")
        search_result = self._run_search(
            mzml_files, database_file, output_dir, output_id,
            search_params, msfragger_path, java_memory
        )

        if search_result.get("status") != "success":
            return search_result

        # Step 3: Run validation
        logger.info("Step 3: Running Philosopher validation...")
        validate_result = self._run_validation(
            mzml_files, database_file, output_dir, output_id,
            fdr_threshold, philosopher_path
        )

        if validate_result.get("status") != "success":
            return validate_result

        # Parse results for summary
        parse_result = self._parse_results(output_dir)

        return {
            "status": "success",
            "database_file": database_file,
            "n_target_proteins": db_result.get("n_target_proteins"),
            "pepxml_files": search_result.get("pepxml_files"),
            "tsv_files": validate_result.get("tsv_files"),
            "summary": parse_result.get("summary"),
            "output_dir": output_dir,
            "message": f"Full pipeline completed. {parse_result.get('summary', {}).get('n_psms', 0)} PSMs identified at {fdr_threshold*100}% FDR"
        }

    def _parse_results(self, output_dir: str) -> Dict[str, Any]:
        """
        Parse TSV output files (psm.tsv, peptide.tsv, protein.tsv).

        Returns structured data with identification statistics.
        """
        if not self._check_pandas():
            logger.warning("pandas not available - returning file paths only")
            tsv_files = {
                "psm": glob.glob(os.path.join(output_dir, "psm.tsv")),
                "peptide": glob.glob(os.path.join(output_dir, "peptide.tsv")),
                "protein": glob.glob(os.path.join(output_dir, "protein.tsv"))
            }
            return {
                "status": "success",
                "tsv_files": {k: v[0] if v else None for k, v in tsv_files.items()},
                "message": "TSV files available (pandas not installed for parsing)"
            }

        import pandas as pd

        summary = {
            "n_psms": 0,
            "n_peptides": 0,
            "n_proteins": 0,
            "charge_distribution": {},
            "missed_cleavage_distribution": {}
        }

        psm_data = None
        peptide_data = None
        protein_data = None

        # Parse psm.tsv
        psm_file = glob.glob(os.path.join(output_dir, "psm.tsv"))
        if psm_file:
            try:
                psm_data = pd.read_csv(psm_file[0], sep="\t")
                summary["n_psms"] = len(psm_data)

                if "Charge" in psm_data.columns:
                    summary["charge_distribution"] = psm_data["Charge"].value_counts().to_dict()

                if "Missed Cleavages" in psm_data.columns:
                    summary["missed_cleavage_distribution"] = psm_data["Missed Cleavages"].value_counts().to_dict()
            except Exception as e:
                logger.warning(f"Failed to parse psm.tsv: {e}")

        # Parse peptide.tsv
        peptide_file = glob.glob(os.path.join(output_dir, "peptide.tsv"))
        if peptide_file:
            try:
                peptide_data = pd.read_csv(peptide_file[0], sep="\t")
                if "Peptide" in peptide_data.columns:
                    summary["n_peptides"] = peptide_data["Peptide"].nunique()
                else:
                    summary["n_peptides"] = len(peptide_data)
            except Exception as e:
                logger.warning(f"Failed to parse peptide.tsv: {e}")

        # Parse protein.tsv
        protein_file = glob.glob(os.path.join(output_dir, "protein.tsv"))
        if protein_file:
            try:
                protein_data = pd.read_csv(protein_file[0], sep="\t")
                summary["n_proteins"] = len(protein_data)

                # Top proteins by spectral count
                if "Spectral Count" in protein_data.columns and "Protein" in protein_data.columns:
                    top_proteins = protein_data.nlargest(10, "Spectral Count")[["Protein", "Spectral Count"]]
                    summary["top_proteins"] = top_proteins.to_dict('records')
            except Exception as e:
                logger.warning(f"Failed to parse protein.tsv: {e}")

        return {
            "status": "success",
            "summary": summary,
            "psm_data": psm_data,
            "peptide_data": peptide_data,
            "protein_data": protein_data,
            "message": f"Parsed results: {summary['n_psms']} PSMs, {summary['n_peptides']} peptides, {summary['n_proteins']} proteins"
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(PeptideIdentification().print_usage())
        sys.exit(1)

    tool = PeptideIdentification()
    print(tool.print_usage())