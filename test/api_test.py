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
            "task": "similar_protein_search_msa",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "similar_protein_search", "search_type": "msa", "protein": "MKFLILLFNILCLFPVLAADNH"}}'
            """
        },
        {
            "task": "similar_protein_search_foldseek",
            "command": f"""
            curl -X 'POST' '{base_url}/run_pipeline/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{{"task": "similar_protein_search", "search_type": "foldseek", "protein": "./checkpoints/server/test_data/4xli_B.pdb"}}'
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
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, timeout=120)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenBioMed API 测试脚本")
    parser.add_argument("--url", type=str, default=DEFAULT_BASE_URL, help="服务地址 (默认: http://127.0.0.1:8090)")
    parser.add_argument("--test", type=str, default=None, help="只运行指定的测试 (如: healthz, molecule)")
    args = parser.parse_args()

    print(f"测试服务地址: {args.url}")
    print("=" * 50)

    run_tests(args.url, args.test)