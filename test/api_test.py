import subprocess
import logging
import argparse

# 配置日志
logging.basicConfig(
    filename="test_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 默认服务地址
DEFAULT_BASE_URL = "http://127.0.0.1:8090"

def get_curl_commands(base_url):
    """生成 curl 命令列表，使用指定的 base_url"""
    return [
        {
            "task": "healthz",
            "command": f"curl -s '{base_url}/healthz'"
        },
        {
            "task": "text_based_molecule_editing",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "text_based_molecule_editing", "model": "molt5", "molecule":"Nc1[nH]c(C(=O)c2ccccc2)c(-c2ccccn2)c1C(=O)c1c[nH]c2ccc(Br)cc12", "text": "This molecule can bind with recombinant human 15-LOX-1"}}'
            """
        },
        {
            "task": "structure_based_drug_design",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "structure_based_drug_design", "model": "pharmolix_fm", "pocket": "./checkpoints/server/test_data/4xli_B_ref_pocket.pkl"}}'
            """
        },
        {
            "task": "molecule_question_answering",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "molecule_question_answering", "model": "molt5", "molecule": "C[C@@H]1CCCCO[C@@H](CN(C)C(=O)Cc2ccccc2)[C@@H](C)CN([C@@H](C)CO)C(=O)c2cc(NS(C)(=O)=O)ccc2O1", "text": "Please identify if this molecule has a role as a conjugate acid, and if so, what is its paired conjugate base?"}}'
            """
        },
        {
            "task": "visualize_molecule",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "visualize_molecule", "visualize": "3D", "molecule": "CC(=O)Oc1ccccc1C(=O)O"}}'
            """
        },
        {
            "task": "visualize_complex",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "visualize_complex", "molecule": "CC(=O)Oc1ccccc1C(=O)O", "protein": "./checkpoints/server/test_data/4xli_B.pdb"}}'
            """
        },
        {
            "task": "molecule_name_request",
            "command": f"""
            curl -X 'POST' '{base_url}/web_search/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "molecule_name_request", "query": "aspirin"}}'
            """
        },
        {
            "task": "molecule_property_prediction",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "molecule_property_prediction", "model": "graphmvp", "dataset": "BBBP", "molecule":"CC(=O)Oc1ccccc1C(=O)O"}}'
            """
        },
        {
            "task": "web_search",
            "command": f"""
            curl -X 'POST' '{base_url}/web_search/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "web_search", "query": "糖尿病"}}'
            """
        },
        {
            "task": "protein_binding_site_prediction",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "protein_binding_site_prediction", "model": "p2pocket", "protein": "./checkpoints/server/test_data/4xli_B.pdb"}}'
            """
        },
        {
            "task": "protein_question_answering",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "protein_question_answering", "model": "biot5", "protein": "MRVGVIRFPGSNCDRDVHHVLELAGAEPEYVWWNQRNLDHLDAVVIPGGFSYGDYLRAGAIAAITPVMDAVRELVREEKPVLGICNGAQILAEVGLVPGVFTVNEHPKFNCQWTELRVKTTRTPFTGLFKKDEVIRMPVAHAEGRYYHDNISEVWENDQVVLQFHGENPNGSLDGITGVCDESGLVCAVMPHPERASEEILGSVDGFKFFRGILKFRG", "text": "Inspect the protein sequence and offer a concise description of its properties."}}'
            """
        },
        {
            "task": "mutation_explanation",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "mutation_explanation", "model": "mutaplm", "mutation": "H163A", "protein": "MTLENVLEAARHLHQTLPALSEFGNWPTDLTATGLQPRAIPATPLVQALDQPGSPRTTGLVQAIRSAAHLAHWKRTYTEAEVGADFRNRYGYFELFGPTGHFHSTQLRGYVAYWGAGLDYDWHSHQAEELYLTLAGGAVFKVDGERAFVGAEGTRLHASWQSAAMSTGDQPILTFVLWRGEGLNALPRMDAA"}}'
            """
        },
        {
            "task": "mutation_engineering",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "mutation_engineering", "model": "mutaplm", "text": "Strongly enhanced InsP6 kinase activity.", "protein": "MASDAAAEPSSGVTHPPRYVIGYALAPKKQQSFIQPSLVAQAASRGMDLVPVDASQPLAEQGPFHLLIHALYGDDWRAQLVAFAARHPAVPIVDPPHAIDRLHNRISMLQVVSELDHAADQDSTFGIPSQVVVYDAAALADFGLLAALRFPLIAKPLVADGTAKSHKMSLVYHREGLGKLRPPLVLQEFVNHGGVIFKVYVVGGHVTCVKRRSLPDVSPEDDASAQGSVSFSQVSNLPTERTAEEYYGEKSLEDAVVPPAAFINQIAGGLRRALGLQLFNFDMIRDVRAGDRYLVIDINYFPGYAKMPGYETVLTDFFWEMVHKDGVGNQQEEKGANHVVVK"}}'
            """
        },
        {
            "task": "pocket_molecule_docking",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "pocket_molecule_docking", "model": "pharmolix_fm", "molecule": "./checkpoints/server/test_data/4xli_B_ref.sdf", "pocket": "./checkpoints/server/test_data/4xli_B_ref_pocket.pkl"}}'
            """
        },
        {
            "task": "protein_molecule_docking_score",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "protein_molecule_docking_score", "model": "vina", "molecule": "./checkpoints/server/test_data/4xli_B_ref.sdf", "protein": "./checkpoints/server/test_data/4xli_B.pdb"}}'
            """
        },
        {
            "task": "protein_folding",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "protein_folding", "model": "esmfold", "protein": "MASDAAAEPSSGVTHPPRYVIGYALAPKKQQSFIQPSLVAQAASRGMDLVPVDASQPLAEQGPFHLLIHALYGDDWRAQLVAFAARHPAVPIVDPPHAIDRLHNRISMLQVVSELDHAADQDSTFGIPSQVVVYDAAALADFGLLAALRFPLIAKPLVADGTAKSHKMSLVYHREGLGKLRPPLVLQEFVNHGGVIFKVYVVGGHVTCVKRRSLPDVSPEDDASAQGSVSFSQVSNLPTERTAEEYYGEKSLEDAVVPPAAFINQIAGGLRRALGLQLFNFDMIRDVRAGDRYLVIDINYFPGYAKMPGYETVLTDFFWEMVHKDGVGNQQEEKGANHVVVK"}}'
            """
        },
        {
            "task": "visualize_protein",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "visualize_protein", "visualize": "cartoon", "protein": "./checkpoints/server/test_data/4xli_B.pdb"}}'
            """
        },
        {
            "task": "visualize_protein_pocket",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "visualize_protein_pocket", "protein": "./checkpoints/server/test_data/4xli_B.pdb", "pocket": "./checkpoints/server/test_data/4xli_B_ref_pocket.pkl"}}'
            """
        },
        {
            "task": "molecule_structure_request",
            "command": f"""
            curl -X 'POST' '{base_url}/web_search/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "molecule_structure_request", "threshold": "0.95", "molecule": "C=C(O)C1CC(C)C23CC4OC(C)COC4C(C)C4CC(O2)C(O)C4C32CC12"}}'
            """
        },
        {
            "task": "protein_uniprot_request",
            "command": f"""
            curl -X 'POST' '{base_url}/web_search/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "protein_uniprot_request", "query": "P0DTC2"}}'
            """
        },
        {
            "task": "protein_pdb_request",
            "command": f"""
            curl -X 'POST' '{base_url}/web_search/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "protein_pdb_request", "query": "6LVN"}}'
            """
        },
        {
            "task": "export_molecule",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "export_molecule", "molecule": "./checkpoints/server/test_data/4xli_B_ref.sdf"}}'
            """
        },
        {
            "task": "export_protein",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "export_protein", "protein": "./checkpoints/server/test_data/4xli_B.pdb"}}'
            """
        },
        {
            "task": "import_pocket",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "import_pocket", "protein": "./checkpoints/server/test_data/4xli_B.pdb", "indices": "76;77;78;79;80"}}'
            """
        },
        {
            "task": "molecule_similarity",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "molecule_similarity", "model": "rdkit", "molecule_1": "CC(=O)Oc1ccccc1C(=O)O", "molecule_2": "CC(=O)Oc1ccccc1C(=O)O"}}'
            """
        },
        {
            "task": "molecule_property_calculation",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "molecule_property_calculation", "model": "rdkit", "molecule": "CC(=O)Oc1ccccc1C(=O)O", "property": "QED"}}'
            """
        },
        {
            "task": "binding_affinity",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "binding_affinity", "protein_complex": "./checkpoints/server/test_data/1avx.pdb"}}'
            """
        },
        {
            "task": "antibody_structure",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "antibody_structure", "heavy_chain": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYYMAWVRQAPGKGLEWVSAISSSGGSTYYADSVKGRLTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYKHNWFGTEVTVELTK", "light_chain": "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPPTFGQGTKVEIK"}}'
            """
        },
        {
            "task": "antibody_design",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "antibody_design", "fasta": "./test_data/design.fasta", "antigen_pdb": "./test_data/antigen.pdb", "epitope": "7 8 9 10 11"}}'
            """
        },
        {
            "task": "similar_protein_search",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "similar_protein_search", "search_type": "foldseek", "protein": "./checkpoints/server/test_data/pdb_6LVN.pdb"}}'
            """
        },
        # tFold antibody structure prediction tests
        {
            "task": "tfold_antibody_structure_antibody",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "tfold_antibody_structure", "prediction_type": "antibody", "heavy_chain": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYYMAWVRQAPGKGLEWVSAISSSGGSTYYADSVKGRLTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYKHNWFGTEVTVELTK", "light_chain": "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPPTFGQGTKVEIK", "output_name": "test_antibody"}}'
            """
        },
        {
            "task": "tfold_antibody_structure_nanobody",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "tfold_antibody_structure", "prediction_type": "nanobody", "heavy_chain": "MSIQEIQKEIAQIQAVIAGIQKYIYTMSIEEIQKQIAAIQCQIAAIQKQIYAMSIEEIQKQIAAIQEQILAIYKQIMAMVT", "output_name": "test_nanobody"}}'
            """
        },
        {
            "task": "tfold_antibody_structure_complex",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "tfold_antibody_structure", "prediction_type": "complex", "heavy_chain": "EVQLVQSGAEVKKPGESLKISCKGSGYSFSNYWIGWVRQMPGKGLEWMGIIDPSNSYTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCARWYYKPFDVWGQGTLVTVSS", "light_chain": "QSVLTQPPSVSGAPGQRVTISCTGSSSNIGSGYDVHWYQQLPGTAPKLLIYGNSKRPSGVPDRFSGSKSGTSASLAITGLQSEDEADYYCASWTDGLSLVVFGGGTKLTVL", "antigen": "RAVPGGSSPAWTQCQQLSQKLCTLAWSAHPLVGHMDLREEDVPHIQCGDGCDPQGLRDNSQFCLQRIHQGLIFYEKLLGSDIFTGEPSLLPDSPVGQLHASLLGLSQLLQPEGHHWETQQIPSLSPSQPWQRLLLRFKILRSLQAFVAVAARVFAHGAATL", "antigen_id": "A", "output_name": "test_complex"}}'
            """
        },
        # IgGM antibody design tests
        {
            "task": "iggm_antibody_design_nanobody",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "iggm_antibody_design", "design_type": "nanobody", "antigen_pdb": "./tmp/pdb_4xli.pdb", "heavy_chain_mask": "QVQLVESGGDLVQSGGSLKLSCAVSXXXXXXXSIGWFRQAPGKEREAVSYSXXXXXXTYYVASVKGRFTISRDNAKNTAYLQMNNLKPEDTGIYYCAAXXXXXXXXXXXXXXXXXXWGQGTQVTVSS", "epitope": "[109,110,111,112,113,114,115,116,117]", "num_samples": 1, "steps": 10, "output_name": "test_nanobody_design"}}'
            """
        },
        {
            "task": "iggm_antibody_design_heavy_light",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "iggm_antibody_design", "design_type": "heavy_light", "antigen_pdb": "./tmp/pdb_4xli.pdb", "heavy_chain_mask": "VQLVESGGGLVQPGGSLRLSCAASXXXXXXXYMNWVRQAPGKGLEWVSVVXXXXXTFYTDSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARXXXXXXXXXXXXXXWGQGTMVTVSS", "light_chain_mask": "DIQMTQSPSSLSASVGDRVSITCXXXXXXXXXXXWYQQKPGKAPKLLISXXXXXXXGVPSRFSGSGSGTDFTLTITSLQPEDFATYYCXXXXXXXXXXXFGGGTKVEIK", "epitope": "[7,8,9,10,11,12,13,14,108,109,110,111,112,113,114,115,116]", "num_samples": 1, "steps": 10, "output_name": "test_heavy_light_design"}}'
            """
        },
        # Boltz-2 structure prediction tests
        {
            "task": "boltz2_affinity_prediction",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "boltz2_structure_prediction", "prediction_type": "affinity", "sequence": "GSHMGSSGMSSGMG", "smiles": "CCO", "output_name": "test_affinity"}}'
            """
        },
        {
            "task": "boltz2_prot_complex_prediction",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "boltz2_structure_prediction", "prediction_type": "prot_complex", "sequence_1": "GSHMGSSGMSSGMG", "sequence_2": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL", "output_name": "test_prot_complex"}}'
            """
        },
        # Mutation design (real BaseCNN fitness oracle) tests
        {
            "task": "mutation_design_aav",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "mutation_design_aav", "num_rounds": 1, "population_size": 6}}'
            """
        },
        {
            "task": "mutation_design_gfp",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "mutation_design_gfp", "num_rounds": 1, "population_size": 6}}'
            """
        },
        # Read CSV file test (requires a CSV file to exist)
        {
            "task": "read_csv_file",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "read_csv_file", "value": "./tmp/mutation_design_aav/aav_mutants_test.csv", "num_rounds": 10}}'
            """
        },
        # Spatial transcriptomics data loading test
        # Note: Requires Visium data directory with filtered_feature_bc_matrix.h5
        # Run test_spatial_transcriptomics_tool.py first to prepare test data via squidpy
        # The test uses a h5ad file created by squidpy's built-in dataset
        {
            "task": "spatial_transcriptomics_loading_note",
            "command": f"""
            echo "Spatial transcriptomics test requires Visium data directory."
            echo "Run: python test/test_spatial_transcriptomics_tool.py to prepare test data"
            echo "Then use: curl -X POST '{base_url}/run_pipeline/' -d '{{\"task\": \"spatial_transcriptomics_loading\", \"value\": \"./tmp/test_visium_data\", \"query\": \"visium\"}}'"
            """
        },
        # Scanpy single-cell RNA-seq analysis test
        # Note: This test uses pbmc3k dataset from scanpy. Run test_scanpy_analysis_tool.py first to prepare test data.
        {
            "task": "scanpy_analysis_load",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "scanpy_analysis", "protein": "./tmp/test_pbmc3k_input.h5ad", "query": "load"}}'
            """
        },
        {
            "task": "scanpy_analysis_qc",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "scanpy_analysis", "protein": "./tmp/test_pbmc3k_input.h5ad", "query": "qc", "num_rounds": 200, "population_size": 3, "diversity_weight": 5.0}}'
            """
        },
        {
            "task": "scanpy_analysis_full_pipeline",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "scanpy_analysis", "protein": "./tmp/test_pbmc3k_input.h5ad", "query": "full_pipeline", "similarity": 0.5, "num_rounds": 200, "population_size": 3, "diversity_weight": 5.0, "max_mutations": 2000, "required_score": 10, "limit": 40}}'
            """
        },
        # CELLxGENE Census query tests
        # Note: Requires cellxgene-census package installed
        {
            "task": "cellxgene_census_summary",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "cellxgene_census_query", "query": "get_summary"}}'
            """
        },
        {
            "task": "cellxgene_census_datasets",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "cellxgene_census_query", "query": "get_datasets", "text": "homo_sapiens"}}'
            """
        },
        {
            "task": "cellxgene_census_obs",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "cellxgene_census_query", "query": "get_obs", "text": "homo_sapiens", "value": "tissue_general == 'lung' and is_primary_data == True"}}'
            """
        },
        {
            "task": "cellxgene_census_var",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "cellxgene_census_query", "query": "get_var", "text": "homo_sapiens", "dataset": "feature_name in ['CD4', 'CD8A', 'CD19']"}}'
            """
        },
        {
            "task": "cellxgene_census_anndata",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "cellxgene_census_query", "query": "get_anndata", "text": "homo_sapiens", "value": "cell_type == 'B cell' and tissue_general == 'blood' and is_primary_data == True", "dataset": "feature_name in ['CD19', 'CD20', 'MS4A1']", "num_rounds": 1000}}'
            """
        },
        # Proteomics data processing tests
        # Note: Requires pyOpenMS and mzML test file. Download test data from:
        # https://github.com/OpenMS/OpenMS-Tests/tree/master/data
        {
            "task": "proteomics_load_note",
            "command": f"""
            echo "Proteomics test requires mzML file. Download test data or provide your own file."
            echo "Test command: curl -X POST '{base_url}/run_pipeline/' -d '{{\"task\": \"proteomics_data_processing\", \"protein\": \"./tmp/test.mzML\", \"query\": \"load\"}}'"
            """
        },
    ]


def run_tests(base_url, test_filter=None):
    """执行测试"""
    curl_commands = get_curl_commands(base_url)

    results = {"success": [], "failed": []}

    for test in curl_commands:
        task = test["task"]

        # 如果指定了测试过滤器，只运行匹配的测试
        if test_filter and test_filter.lower() not in task.lower():
            continue

        command = test["command"].strip()

        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, timeout=180)
            output = result.stdout.strip()
            logging.info(f"Task: {task} - Success: {output}")
            print(f"[PASS] Task: {task}")
            print(f"       Output: {output[:100]}...")
            results["success"].append(task)
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.strip() or e.stdout.strip()
            logging.error(f"Task: {task} - Error: {error_message}")
            print(f"[FAIL] Task: {task}")
            print(f"       Error: {error_message[:200]}")
            results["failed"].append({"task": task, "error": error_message})
        except subprocess.TimeoutExpired:
            logging.error(f"Task: {task} - Timeout")
            print(f"[TIMEOUT] Task: {task}")
            results["failed"].append({"task": task, "error": "Timeout"})
        except Exception as e:
            logging.error(f"Task: {task} - Unexpected error: {str(e)}")
            print(f"[ERROR] Task: {task} - {str(e)}")
            results["failed"].append({"task": task, "error": str(e)})

    # 打印总结
    print("\n" + "=" * 50)
    print(f"测试总结 (base_url: {base_url})")
    print("=" * 50)
    print(f"成功: {len(results['success'])}")
    print(f"失败: {len(results['failed'])}")
    if results['failed']:
        print("\n失败的接口:")
        for f in results['failed']:
            print(f"  - {f['task']}: {f['error'][:100]}")

    return results


def run_upload_tests(base_url, upload_key):
    """测试 /api/upload 接口"""
    print("\n" + "=" * 50)
    print("Upload API Tests")
    print("=" * 50)

    test_file = "./checkpoints/server/test_data/4xli_B.pdb"
    results = {"success": [], "failed": []}

    # Test 1: 正常上传 PDB 文件
    task = "upload_pdb_valid_key"
    cmd = f"curl -s -X POST '{base_url}/api/upload' -H 'X-API-Key: {upload_key}' -F 'file=@{test_file}'"
    print(f"\n[1] Upload PDB file with valid key...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        if '"path"' in output and '"filename"' in output:
            logging.info(f"Task: {task} - Success: {output}")
            print(f"[PASS] {task}")
            print(f"       Response: {output}")
            results["success"].append(task)
        else:
            logging.error(f"Task: {task} - Unexpected response: {output}")
            print(f"[FAIL] {task}")
            print(f"       Response: {output[:200]}")
            results["failed"].append({"task": task, "error": f"Unexpected response: {output[:200]}"})
    except Exception as e:
        logging.error(f"Task: {task} - Error: {str(e)}")
        print(f"[FAIL] {task} - {str(e)}")
        results["failed"].append({"task": task, "error": str(e)})

    # Test 2: 不带 API Key 上传（应返回 422：FastAPI Header 参数校验先于业务逻辑）
    task = "upload_no_key"
    cmd = f"curl -s -w '\\n%{{http_code}}' -X POST '{base_url}/api/upload' -F 'file=@{test_file}'"
    print(f"\n[2] Upload without API key (expect 422)...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        lines = result.stdout.strip().split('\n')
        http_code = lines[-1].strip() if lines else ""
        body = '\n'.join(lines[:-1]) if len(lines) > 1 else result.stdout.strip()
        if http_code == "422":
            logging.info(f"Task: {task} - Success: returned 422 as expected")
            print(f"[PASS] {task} (returned 422: missing required header)")
            print(f"       Response: {body[:100]}")
            results["success"].append(task)
        else:
            logging.error(f"Task: {task} - Expected 422, got {http_code}: {body}")
            print(f"[FAIL] {task} (expected 422, got {http_code})")
            print(f"       Response: {body[:200]}")
            results["failed"].append({"task": task, "error": f"Expected 422, got {http_code}: {body[:200]}"})
    except Exception as e:
        logging.error(f"Task: {task} - Error: {str(e)}")
        print(f"[FAIL] {task} - {str(e)}")
        results["failed"].append({"task": task, "error": str(e)})

    # Test 3: 错误的 API Key（应返回 401）
    task = "upload_wrong_key"
    cmd = f"curl -s -w '\\n%{{http_code}}' -X POST '{base_url}/api/upload' -H 'X-API-Key: wrong_key_123' -F 'file=@{test_file}'"
    print(f"\n[3] Upload with wrong API key (expect 401)...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        lines = result.stdout.strip().split('\n')
        http_code = lines[-1].strip() if lines else ""
        body = '\n'.join(lines[:-1]) if len(lines) > 1 else result.stdout.strip()
        if http_code == "401":
            logging.info(f"Task: {task} - Success: returned 401 as expected")
            print(f"[PASS] {task} (returned 401)")
            print(f"       Response: {body[:100]}")
            results["success"].append(task)
        else:
            logging.error(f"Task: {task} - Expected 401, got {http_code}: {body}")
            print(f"[FAIL] {task} (expected 401, got {http_code})")
            print(f"       Response: {body[:200]}")
            results["failed"].append({"task": task, "error": f"Expected 401, got {http_code}: {body[:200]}"})
    except Exception as e:
        logging.error(f"Task: {task} - Error: {str(e)}")
        print(f"[FAIL] {task} - {str(e)}")
        results["failed"].append({"task": task, "error": str(e)})

    # Test 4: 上传不支持的文件类型（应返回 400）
    task = "upload_unsupported_type"
    cmd = f"curl -s -w '\\n%{{http_code}}' -X POST '{base_url}/api/upload' -H 'X-API-Key: {upload_key}' -F 'file=@./test/api_test.py'"
    print(f"\n[4] Upload unsupported file type (.py) (expect 400)...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        lines = result.stdout.strip().split('\n')
        http_code = lines[-1].strip() if lines else ""
        body = '\n'.join(lines[:-1]) if len(lines) > 1 else result.stdout.strip()
        if http_code == "400":
            logging.info(f"Task: {task} - Success: returned 400 as expected")
            print(f"[PASS] {task} (returned 400)")
            print(f"       Response: {body[:100]}")
            results["success"].append(task)
        else:
            logging.error(f"Task: {task} - Expected 400, got {http_code}: {body}")
            print(f"[FAIL] {task} (expected 400, got {http_code})")
            print(f"       Response: {body[:200]}")
            results["failed"].append({"task": task, "error": f"Expected 400, got {http_code}: {body[:200]}"})
    except Exception as e:
        logging.error(f"Task: {task} - Error: {str(e)}")
        print(f"[FAIL] {task} - {str(e)}")
        results["failed"].append({"task": task, "error": str(e)})

    # Print summary
    print("\n" + "-" * 30)
    print(f"Upload Tests: {len(results['success'])} passed, {len(results['failed'])} failed")
    if results['failed']:
        print("Failed:")
        for f in results['failed']:
            print(f"  - {f['task']}: {f['error'][:100]}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenBioMed API 测试脚本")
    parser.add_argument("--url", type=str, default=DEFAULT_BASE_URL, help="服务地址 (默认: http://127.0.0.1:8090)")
    parser.add_argument("--test", type=str, default=None, help="只运行指定的测试 (如: healthz, molecule)")
    parser.add_argument("--upload-key", type=str, default=None, help="Upload API key (提供后自动运行 upload 测试)")
    args = parser.parse_args()

    print(f"测试服务地址: {args.url}")
    print("=" * 50)

    run_tests(args.url, args.test)

    if args.upload_key:
        run_upload_tests(args.url, args.upload_key)
    else:
        print("\n[提示] 如需测试 /api/upload 接口，请使用 --upload-key 参数，例如:")
        print("  python test/api_test.py --url http://127.0.0.1:8095 --upload-key your_key")