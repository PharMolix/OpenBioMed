#!/usr/bin/env python3
import json
import os
import urllib.parse
from datetime import datetime

STATE_FILE = "route_state.json"
OUTPUT_FILE = "tree_visualization.html"

def get_svg_data_uri(smiles):
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Set cleaner, high-contrast flat rendering for scientific clarity
            d2d = Draw.MolDraw2DSVG(300, 200)
            opt = d2d.drawOptions()
            opt.addStereoAnnotation = True
            opt.clearBackground = False  # Allows transparent backgrounds in the nodes
            opts = d2d.drawOptions()
            opts.useBWAtomPalette() # Use black and white palette for serious scientific feel, less "flashy"
            d2d.DrawMolecule(mol)
            d2d.FinishDrawing()
            svg_text = d2d.GetDrawingText()
            return svg_text # returning raw SVG string for injecting directly in DOM rather than DataURI scaling issues
    except Exception:
        pass
    return "<i>[Structure Render Failed]</i>"

def get_status_box(val, is_root=False):
    is_solved = val.get("solved", False)
    is_purchasable = val.get("purchasable", False)
    
    if is_purchasable:
        bgcolor, color, text = "#d1e7dd", "#0f5132", "🛒 Purchasable & Solved"
    elif is_solved:
        bgcolor, color, text = "#cff4fc", "#055160", "🟢 Fully Synthesizable"
    else:
        bgcolor, color, text = "#f8d7da", "#842029", "🛑 Unresolved / Needs Splitting"
        
    border = f"3px solid {color}" if is_root else "1px solid #dee2e6"
    return bgcolor, color, text, border

def render():
    if not os.path.exists(STATE_FILE):
        print(f"Error: {STATE_FILE} not found. Initialize a tree first.")
        return
        
    with open(STATE_FILE, "r") as f:
        t = json.load(f)
        
    root_id = t.get('root', '')
    if not root_id: return
    
    # We will build a clean nested HTML list (ul/li) hierarchy 
    # to represent the AND/OR tree downwards natively.
    def build_node_html(node_id, depth=0):
        node = t['nodes'][node_id]
        if node['type'] == 'molecule':
            bg_color, font_color, status_text, border = get_status_box(node, is_root=(node_id == root_id))
            svg = get_svg_data_uri(node['smiles'])
            
            html = f"""
            <div class="mol-card" style="background-color: {bg_color}; border: {border};">
                <div class="mol-header">
                    <strong>{node_id}</strong> &mdash; 
                    <span style="color: {font_color}; font-weight: bold; font-size: 0.85em;">{status_text}</span>
                </div>
                <div class="mol-body">
                    <div class="mol-svg">{svg}</div>
                    <div class="mol-smiles"><code>{node['smiles']}</code></div>
                </div>
            </div>
            """
            
            if len(node.get('reactions', [])) > 0:
                html += f"<ul>"
                for rxn_id in node['reactions']:
                    html += f"<li>{build_node_html(rxn_id, depth+1)}</li>"
                html += f"</ul>"
                
            return html
            
        elif node['type'] == 'reaction':
            is_solved = node.get("solved", False)
            icon = "✅" if is_solved else "⏳"
            color = "#198754" if is_solved else "#6c757d"
            
            html = f"""
            <div class="rxn-card">
                <div style="color: {color}; font-weight: 600; font-size: 0.9em; margin-bottom: 5px;">
                    {icon} Reaction: {node.get('name', 'Unknown')} (<i>{node_id}</i>)
                </div>
            </div>
            """
            
            if len(node.get('children', [])) > 0:
                html += f"<ul>"
                for child_id in node['children']:
                    html += f"<li>{build_node_html(child_id, depth+1)}</li>"
                html += f"</ul>"
                
            return html

    tree_html = f"<ul><li>{build_node_html(root_id)}</li></ul>"

    page_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Retrosynthetic Pathway Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f8f9fa;
            color: #212529;
            line-height: 1.5;
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{ border-bottom: 2px solid #dee2e6; padding-bottom: 10px; margin-bottom: 30px; font-weight: 500; font-size: 1.5rem; }}
        
        /* Clean hierarchy representation via ul/li */
        ul {{ list-style-type: none; padding-left: 40px; margin: 15px 0; border-left: 2px dashed #ced4da; }}
        li {{ position: relative; margin: 25px 0; }}
        li::before {{
            content: ''; position: absolute; left: -42px; top: 20px;
            width: 32px; height: 2px; background-color: #ced4da;
        }}
        
        /* Molecule Cards */
        .mol-card {{
            display: inline-block;
            background: #fff;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 15px;
            min-width: 300px;
            max-width: 450px;
            vertical-align: top;
        }}
        .mol-header {{ font-size: 0.9em; margin-bottom: 12px; border-bottom: 1px solid rgba(0,0,0,0.1); padding-bottom: 8px; }}
        .mol-svg {{ text-align: center; margin: 10px 0; background: #fff; padding: 10px; border-radius: 4px; }}
        .mol-smiles {{
            font-size: 0.75rem; 
            background: #e9ecef; 
            padding: 6px 10px; 
            border-radius: 4px; 
            word-break: break-all;
            color: #495057;
        }}
        
        /* Reaction Cards */
        .rxn-card {{
            display: inline-block;
            padding: 8px 15px;
            background: #e9ecef;
            border-left: 4px solid #adb5bd;
            border-radius: 0 4px 4px 0;
            font-size: 0.95em;
            margin-top: 5px;
        }}
        
        .meta {{ font-size: 0.8em; color: #6c757d; margin-bottom: 40px; }}
    </style>
</head>
<body>
    <h1>Retrosynthetic Synthesis State Tree</h1>
    <div class="meta">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Mode: Static Hierarchical View</div>
    
    <div>
        {tree_html}
    </div>
</body>
</html>
"""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(page_html)
    print(f"Success! Rendered crisp, readable schematic HTML tree to {OUTPUT_FILE}")

if __name__ == "__main__":
    render()
