import sys
import argparse
import urllib.request
import urllib.parse
import json

def run_aizynthfinder_api(target_smiles):
    try:
        from aizynthfinder.aizynthfinder import AiZynthFinder
        import os
        import platform

        # Assume models are downloaded via `download_public_data <DATA_DIR>`
        data_dir = os.environ.get("AIZYNTH_DATA_DIR", "C:/tmp" if platform.system() == "Windows" else "/tmp")
        model_path = os.path.join(data_dir, "uspto_model.onnx")
        templates_path = os.path.join(data_dir, "uspto_templates.csv.gz")

        if not os.path.exists(model_path) or not os.path.exists(templates_path):
            print(json.dumps({"error": f"Model files missing in {data_dir}. Please run `download_public_data {data_dir}` first."}))
            return

        config_dict = {
            "expansion": {
                "uspto": [model_path, templates_path]
            },
            "search": {
                "time_limit": 10,
                "iteration_limit": 10
            }
        }
        
        finder = AiZynthFinder(configdict=config_dict)
        finder.expansion_policy.select("uspto")
        finder.target_smiles = target_smiles
        finder.tree_search(show_progress=False)
        finder.build_routes()
        routes = getattr(finder.routes, "dicts", getattr(finder.routes, "dictionaries", []))
        
        results = []
        for route in routes[:10]:
            if "children" in route:
                for child in route["children"]:
                    # Get immediate precursors
                    precursors = [c["smiles"] for c in child.get("children", [])]
                    if precursors:
                        results.append({
                            "reaction_description": child.get("metadata", {}).get("type", "Unknown Reaction"),
                            "precursors": precursors,
                            "score": route.get("score", 0.5)
                        })
        
        if not results:
            print(json.dumps({"status": "failed", "message": "No results matched the target molecule from AiZynthFinder."}))
        else:
            print(json.dumps({"status": "success", "data": {"results": results}}, indent=2))
            
    except ImportError:
        print(json.dumps({"error": "aizynthfinder is not installed. Please run `pip install aizynthfinder`."}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Failed to run AiZynthFinder locally. {str(e)}"}))
        sys.exit(1)

def check_commercial(smiles):
    # Use PubChem REST API to check if the compound exists
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{urllib.parse.quote(smiles)}/cids/JSON"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        data = json.loads(response.read().decode('utf-8'))
        cids = data.get("IdentifierList", {}).get("CID", [])
        if cids:
            return True
    except Exception as e:
        pass
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query AiZynthFinder Engine or Vendor Catalogs")
    parser.add_argument("--retro", type=str, help="Target SMILES to split via local AiZynthFinder")
    parser.add_argument("--vendor", type=str, help="SMILES to check for commercial availability via PubChem")
    args = parser.parse_args()

    if args.retro:
        run_aizynthfinder_api(args.retro)
    elif args.vendor:
        is_avail = check_commercial(args.vendor)
        print(json.dumps({
            "query": args.vendor,
            "status": "resolved" if is_avail else "unresolved",
            "is_purchasable_proxy": is_avail
        }, indent=2))
    else:
        parser.print_help()
