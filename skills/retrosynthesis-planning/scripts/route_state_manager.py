import json
import argparse
import sys
import os

STATE_FILE = "route_state.json"

def load_tree():
    if not os.path.exists(STATE_FILE): return None
    with open(STATE_FILE, 'r') as f: return json.load(f)

def save_tree(t):
    with open(STATE_FILE, 'w') as f: json.dump(t, f, indent=4)

def init_tree(smiles):
    t = {
        "root": "M0", 
        "nodes": {
            "M0": {
                "id": "M0", 
                "type": "molecule", 
                "smiles": smiles, 
                "purchasable": False, 
                "solved": False, 
                "reactions": [], 
                "visits": 1
            }
        }, 
        "c_m": 1, 
        "c_r": 0
    }
    save_tree(t)
    print(f"Tree initialized with explicit root M0 (Molecule Node): {smiles}")

def backpropagate():
    t = load_tree()
    if not t: return
    
    # Ground-up re-evaluation (Supports pruning and fallbacks)
    for nid, node in t['nodes'].items():
        if node['type'] == 'molecule':
            node['solved'] = node.get('purchasable', False)
        elif node['type'] == 'reaction':
            node['solved'] = False
            
    changed = True
    while changed:
        changed = False
        for nid, node in t['nodes'].items():
            if node['type'] == 'reaction' and not node['solved']:
                if all(t['nodes'][c]['solved'] for c in node['children']):
                    node['solved'] = True
                    changed = True
            elif node['type'] == 'molecule' and not node['solved']:
                if any(t['nodes'][r].get('solved', False) for r in node.get('reactions', [])):
                    node['solved'] = True
                    changed = True
                    
    save_tree(t)
    root_solved = t['nodes'][t['root']]['solved']
    print(f"\n[Backprop] Cycle complete. Root molecule solved status: {root_solved}\n")

def expand_node(parent_id, rxn_name, children_smiles, purchasable_list):
    t = load_tree()
    if not t:
        print("Error: Must init tree first.")
        sys.exit(1)
        
    if parent_id not in t['nodes'] or t['nodes'][parent_id]['type'] != 'molecule':
        print(f"Error: Invalid parent ID ({parent_id}). Must be an existing molecule OR-node ID.")
        sys.exit(1)
        
    # Global Molecule Deduplication
    smiles_to_mid = {node['smiles']: k for k, node in t['nodes'].items() if node['type'] == 'molecule'}
    
    child_ids = []
    updated_mids = []
    for sm, p_str in zip(children_smiles, purchasable_list):
        is_p = p_str.lower() == 'true'
        
        if sm in smiles_to_mid:
            mid = smiles_to_mid[sm]
            child_ids.append(mid)
            if is_p and not t['nodes'][mid]['purchasable']:
                t['nodes'][mid]['purchasable'] = True
                updated_mids.append(mid)
        else:
            mid = f"M{t['c_m']}"; t['c_m'] += 1
            t['nodes'][mid] = {
                "id": mid, 
                "type": "molecule", 
                "smiles": sm, 
                "purchasable": is_p, 
                "solved": is_p, 
                "reactions": [], 
                "visits": 1
            }
            child_ids.append(mid)
            smiles_to_mid[sm] = mid
            
    # Duplicate Reaction check on this parent
    existing_reaction = False
    for r_id in t['nodes'][parent_id].get('reactions', []):
        r_node = t['nodes'][r_id]
        if r_node['name'] == rxn_name and set(r_node['children']) == set(child_ids):
            existing_reaction = True
            print(f"Reaction [{r_id}] already exists on [{parent_id}]. Skipping duplicate insertion.")
            break
            
    if not existing_reaction:
        rid = f"R{t['c_r']}"; t['c_r'] += 1
        t['nodes'][rid] = {
            "id": rid, 
            "type": "reaction", 
            "name": rxn_name, 
            "children": child_ids, 
            "solved": False
        }
        t['nodes'][parent_id].setdefault('reactions', []).append(rid)
        print(f"Added Reaction mapped as AND-Node [{rid}] to parent [{parent_id}]. Generated/linked fragment children: {child_ids}")
    
    if updated_mids:
        print(f"Updated purchasability for existing fragments: {updated_mids}")
        
    save_tree(t)
    backpropagate()

def show_status():
    t = load_tree()
    if not t:
        print("Tree is empty or not initialized.")
        return
    root_id = t['root']
    root_node = t['nodes'][root_id]
    print(f"--- MCTS Tree Status ---")
    print(f"Target Root: [{root_id}] {root_node['smiles']}")
    print(f"Overall Solved: {root_node.get('solved', False)}")
    
    unsolved_leaves = []
    for nid, node in t['nodes'].items():
        if node['type'] == 'molecule' and not node['solved'] and len(node.get('reactions', [])) == 0:
            unsolved_leaves.append((nid, node['smiles']))
            
    print(f"\nFrontier (Unsolved Leaf Nodes ready for expansion): {len(unsolved_leaves)}")
    for nid, sm in unsolved_leaves:
        print(f" - [{nid}] {sm}")

def prune_reaction(rxn_id):
    t = load_tree()
    if not t: return
    
    if rxn_id not in t['nodes'] or t['nodes'][rxn_id]['type'] != 'reaction':
        print(f"Error: Reaction {rxn_id} not found.")
        sys.exit(1)
        
    # Remove reference from parents
    for nid, node in t['nodes'].items():
        if node['type'] == 'molecule' and rxn_id in node.get('reactions', []):
            node['reactions'].remove(rxn_id)
            
    # Delete the reaction
    del t['nodes'][rxn_id]
    save_tree(t)
    print(f"Reaction [{rxn_id}] pruned. Recalculating tree state...")
    backpropagate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage MCTS Retrosynthetic AND/OR Tree (Backprop Enabled)")
    parser.add_argument("--init", type=str, help="Initialize tree with target SMILES root")
    parser.add_argument("--expand", type=str, help="Parent molecule ID to expand (e.g. M0)")
    parser.add_argument("--rxn", type=str, help="Name of the chemical reaction (AND-Node)")
    parser.add_argument("--children", nargs='+', help="List of child fragments SMILES")
    parser.add_argument("--purchasable", nargs='+', help="List of 'True'/'False' matching children")
    parser.add_argument("--backprop", action="store_true", help="Force backpropagation of solved state entirely")
    parser.add_argument("--status", action="store_true", help="Show current unsolved leaf nodes (Agent Memory Recall)")
    parser.add_argument("--prune", type=str, help="Prune a reaction branch by Reaction ID (e.g. R0) for fallback/backtracking")
    
    args = parser.parse_args()
    
    if args.init:
        init_tree(args.init)
    elif args.expand:
        if not args.rxn or not args.children or not args.purchasable or len(args.children) != len(args.purchasable):
            print("Error: --rxn, --children, and --purchasable are required and lists must be same length.")
            sys.exit(1)
        expand_node(args.expand, args.rxn, args.children, args.purchasable)
    elif args.status:
        show_status()
    elif args.prune:
        prune_reaction(args.prune)
    elif args.backprop:
        backpropagate()
    else:
        parser.print_help()
