from typing import List, Optional, Union, Callable
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
work_dir = os.path.abspath(__file__).replace("open_biomed/tools/visualization_tools.py", "")

import argparse
from datetime import datetime
import shutil
from rdkit.Chem import Draw, rdDepictor
import subprocess

from open_biomed.tools.base_tool import Tool, serial_exec
from open_biomed.data import Molecule, Protein, Pocket
from open_biomed.utils.config import Config, merge_config
from open_biomed.utils.misc import create_tool_input


def convert_png2gif(png_dir, gif_file, fps=0.5):
    """
    generate gif from png
    :param png_file
    :param gif_file
    """
    import imageio
    frames = [imageio.v2.imread(os.path.join(png_dir, f)) for f in sorted(os.listdir(png_dir)) if f.endswith(".png")]
    imageio.mimsave(gif_file, frames, fps=fps)

def visualize_complex_3D(
    file: str, 
    protein_file: Optional[str]=None, 
    ligand_file: Optional[str]=None,
    config: Config=None,
    rotate=False,
    num_frames=20
):
    try:
        from pymol import cmd
        cmd.reinitialize()
        cmd.bg_color(getattr(config, "background_color", "white"))
        cmd.set("ray_opaque_background", getattr(config, "ray_opaque_background", 1))
        
        if ligand_file is not None:
            cmd.load(ligand_file, "ligand")
            # 防御性编程：检查 molecule.show 属性是否存在
            if hasattr(config.molecule, 'show'):
                for elem in config.molecule.show:
                    cmd.show(elem, "ligand")
            for elem in config.molecule.__dict__.keys():
                if elem not in ["show", "mode"]:
                    cmd.set(elem, config.molecule.__dict__[elem], "ligand")
            # cmd.orient("ligand")
        if protein_file is not None:
            cmd.load(protein_file, "protein")
            cmd.hide("everything", "protein")
            # 防御性编程：检查 protein.show 属性是否存在
            if hasattr(config.protein, 'show'):
                for elem in config.protein.show:
                    cmd.show(elem, "protein")
            for elem in config.protein.__dict__.keys():
                if elem not in ["color", "show", "cnc"]:
                    cmd.set(elem, config.protein.__dict__[elem], "protein")
            protein_color = getattr(config.protein, "color", "grey")
            if protein_color == "spectrum":
                cmd.spectrum("count", selection="protein")
            else:
                cmd.color(protein_color, "protein")
                if getattr(config.protein, "cnc", False):
                    cmd.util.cnc("protein")
            #cmd.orient("protein")

        cmd.zoom("all")
        
        if rotate:
            cmd.mset(f"1 x{num_frames}")
            cmd.util.mroll(1, num_frames, 360 // num_frames)

            name_dir = os.path.dirname(file)
            name_base = os.path.basename(file)
            name_time = datetime.now()
            name_temp = os.path.join(name_dir, "rotate_png_"+name_time.strftime("%Y%m%d_%H%M%S")+"_"+name_base[:-4])
            if not os.path.exists(name_temp):
                os.makedirs(name_temp)
            cmd.mpng(f"{name_temp}/", width=config.width, height=config.height)

            convert_png2gif(name_temp, file)
            if os.path.exists(name_temp):
                shutil.rmtree(name_temp)
        else:
            cmd.png(file, width=config.width, height=config.height, dpi=config.dpi)
    except ImportError:
        print("Please intall PyMol to enable 3D visualization!")

def visualize_protein_with_pocket(
    file: str,
    protein_file: str=None, 
    pocket_indices: List[int]=None,
    config: Config=None,
    rotate=False,
    num_frames=50
) -> str:
    try:
        from pymol import cmd
        cmd.reinitialize()
        cmd.bg_color(getattr(config, "background_color", "white"))
        cmd.set("ray_opaque_background", getattr(config, "ray_opaque_background", 1))
        
        cmd.load(protein_file, "protein")
        cmd.hide("everything", "protein")
        cmd.show("surface", "protein")
        residues = "+".join([str(elem) for elem in pocket_indices])
        cmd.select("highlight", f"protein and resi {residues}")
        
        # Color the selected residues with the chosen highlight color
        cmd.color("red", "highlight")
        cmd.color("grey", "protein and not highlight")
        
        cmd.set("transparency", 0.3, "protein and not highlight")

        if rotate:
            cmd.mset(f"1 x{num_frames}")
            cmd.util.mroll(1, num_frames, 360 // num_frames)

            name_dir = os.path.dirname(file)
            name_base = os.path.basename(file)
            name_time = datetime.now()
            name_temp = os.path.join(name_dir, "protein_pocket_"+name_time.strftime("%Y%m%d_%H%M%S")+"_"+name_base[:-4])
            if not os.path.exists(name_temp):
                os.makedirs(name_temp)
            cmd.mpng(f"{name_temp}/", width=config.width, height=config.height)

            convert_png2gif(name_temp, file)
            if os.path.exists(name_temp):
                shutil.rmtree(name_temp)
        else:
            cmd.png(file, width=config.width, height=config.height, dpi=config.dpi)
    except:
        print("Please Install PyMol to enable 3D visualization!")
    return file

class Visualizer(Tool):
    def __init__(self) -> None:
        super().__init__()

class MoleculeVisualizer(Visualizer):
    def __init__(self) -> None:
        pass

    def print_usage(self) -> str:
        return "\n".join([
            'Visualize molecule.',
            'Inputs: {"molecule": Molecule (an OpenBioMed Molecule object), "rotate": bool (whether to rotate the molecule), "config": str (the visualization style, currently supported: "2D", "ball_and_stick")}',
            "Outputs: str (the path of the generated png figure)."
        ])

    def run(self, 
        molecule: Molecule, 
        config: Optional[Union[str, Config]]=None, 
        img_file: Optional[str]=None,
        rotate: bool=False
    ) -> Union[List[str], List[str]]:
        # img_file_type = "gif" if rotate else "png"
        molecule._add_name()
        if img_file is None:
            img_file_type = "png"
            img_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tmp", f"{molecule.name}.{img_file_type}")
        if config is None:
            config = "2D"
        if isinstance(config, str):
            cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "visualization", "molecule", f"{config}.yaml")
            config = merge_config(
                Config(cfg_path),
                Config(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "visualization", "global_config.yaml"))
            )
        if config.molecule.mode == "3D":
            sdf_file = molecule.save_sdf(overwrite=True)
            visualize_complex_3D(img_file, ligand_file=sdf_file, config=config, rotate=rotate)
        if config.molecule.mode == "2D":
            molecule._add_rdmol()
            rdDepictor.Compute2DCoords(molecule.rdmol)
            Draw.MolToImageFile(molecule.rdmol, img_file, size=(config.width, config.height))

        return [os.path.abspath(img_file)], [os.path.abspath(img_file)]

class ProteinVisualizer(Visualizer):
    def __init__(self) -> None:
        pass

    def print_usage(self) -> str:
        return "\n".join([
            'Visualize protein',
            'Inputs: {"protein": Protein (an OpenBioMed Protein object, 3D structure is required), "rotate": bool (whether to rotate the protein, default: False), "config": str (the visualization style, currently supported: "cartoon", "all_atom", "surface". Default: "cartoon")}',
            "Outputs: str (the path of the generated png figure)."
        ])

    def run(self, 
        protein: Protein, 
        config: Optional[Union[str, Config]]=None, 
        img_file: Optional[str]=None,
        rotate: bool=False,
    ) -> Union[List[str], List[str]]:
        pdb_file = protein.save_pdb(overwrite=True)
        
        if img_file is None:
            # img_file_type = "gif" if rotate else "png"
            img_file_type = "png"
            img_file = f"./tmp/{protein.name}.{img_file_type}"

        if config is None:
            config = "cartoon"
        if isinstance(config, str):
            cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "visualization", "protein", f"{config}.yaml")
            config = merge_config(
                Config(cfg_path),
                Config(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "visualization", "global_config.yaml"))
            )

        visualize_complex_3D(img_file, protein_file=pdb_file, config=config, rotate=rotate)
        os.system(f"rm {os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tmp', 'protein_to_visualize.pdb')}")
        return [os.path.abspath(img_file)], [os.path.abspath(img_file)]

class ComplexVisualizer(Visualizer):
    def __init__(self) -> None:
        pass

    def print_usage(self) -> str:
        return "\n".join([
            'Visualize a ligand-receptor complex.',
            'Inputs: {"molecule": Molecule (an OpenBioMed Molecule object), "protein": Protein (an OpenBioMed Protein object, 3D structure is required), "rotate": bool (whether to create a gif animation by rotating the complex, default: True), "molecule_config": str (the visualization style for the molecule, currently supported: "ball_and_stick". Default: "ball_and_stick"), "protein_config": str (the visualization style for the protein, currently supported: "cartoon", "all_atom", "surface". Default: "cartoon")}',
            "Outputs: str (the path of the generated png figure)."
        ])

    def run(self, 
        molecule: Molecule, 
        protein: Protein, 
        molecule_config: Optional[Union[str, Config]]=None, 
        protein_config: Optional[Union[str, Config]]=None, 
        img_file: Optional[str]=None,
        rotate: bool=True
    ) -> Union[List[str], List[str]]:
        # img_file_type = "gif" if rotate else "png"
        if img_file is None:
            img_file_type = "png"
            img_file = f"./tmp/complex_{molecule.name}_{protein.name}.{img_file_type}"
        sdf_file = molecule.save_sdf(overwrite=True)
        pdb_file = protein.save_pdb(overwrite=True)

        if molecule_config is None:
            molecule_config = "ball_and_stick"
        if isinstance(molecule_config, str):
            cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "visualization", "molecule", f"{molecule_config}.yaml")
            molecule_config = Config(cfg_path)
        if protein_config is None:
            protein_config = "cartoon"
        if isinstance(protein_config, str):
            cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "visualization", "protein", f"{protein_config}.yaml")
            protein_config = Config(cfg_path)
        config = merge_config(merge_config(molecule_config, protein_config), Config(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "visualization", "global_config.yaml")))

        visualize_complex_3D(img_file, ligand_file=sdf_file, protein_file=pdb_file, config=config, rotate=rotate)

        return [os.path.abspath(img_file)], [os.path.abspath(img_file)]

class ProteinPocketVisualizer(Visualizer):
    def __init__(self) -> None:
        super().__init__()

    def print_usage(self) -> str:
        return "\n".join([
            'Visualize pockets in protein.',
            'Inputs: {"protein": the protein, "pocket": the pocket (should be part of the protein), "rotate": whether to rotate the molecule}',
            "Outputs: A figure."
        ])

    def run(self,
        protein: Protein,
        pocket: Pocket,
        img_file: Optional[str]=None,
        rotate: bool=True
    ) -> Union[List[str], List[str]]:
        pdb_file = protein.save_pdb("./tmp/protein_to_visualize.pdb", overwrite=True)
        if img_file is None:
            img_file = f"./tmp/pocket_{protein.name}_{pocket.name}.png"
        print(pdb_file, pocket.orig_indices)
        visualize_protein_with_pocket(img_file, pdb_file, pocket.orig_indices, config=Config("./configs/visualization/global_config.yaml"), rotate=rotate, num_frames=20)
        return [os.path.abspath(img_file)], [os.path.abspath(img_file)]

class PyMolVisualizerWrapper(Tool):
    def __init__(self, task: str) -> None:
        self.task = task
        if task == "visualize_molecule":
            self.visualizer = MoleculeVisualizer()
        elif task == "visualize_protein":
            self.visualizer = ProteinVisualizer()
        elif task == "visualize_complex":
            self.visualizer = ComplexVisualizer()
        elif task == "visualize_protein_pocket":
            self.visualizer = ProteinPocketVisualizer()
        else:
            raise ValueError(f"Invalid task: {task}")

    def print_usage(self) -> str:
        return self.visualizer.print_usage()

    def run_single(self, *args, **kwargs) -> Union[List[str], List[str]]:
        vis_process = [
            "python3", os.path.join(work_dir, "open_biomed/tools/visualization_tools.py"), 
            "--task", self.task,
            "--save_output_filename", os.path.join(work_dir, "tmp/visualization_file.txt"),
        ]
        for key, value in kwargs.items():
            if key in ["molecule", "protein", "pocket"]:
                vis_process.append(f"--{key}")
                if key == "molecule":
                    vis_process.append(value.save_sdf())
                elif key == "protein":
                    vis_process.append(value.save_pdb())
                elif key == "pocket":
                    vis_process.append(value.save_binary())
            elif key == "rotate":
                vis_process.append("--rotate")
            elif key in ["molecule_config", "protein_config", "output_file", "save_output_filename"]:
                vis_process.append(f"--{key}")
                vis_process.append(value)
        subprocess.Popen(vis_process).communicate()
        output = open(os.path.join(work_dir, "tmp/visualization_file.txt"), "r").read()
        output = output, f"The generated figure is saved at {output}"
        return output

    @serial_exec
    def run(self, *args, **kwargs) -> Union[List[str], List[str]]:
        return self.run_single(*args, **kwargs)

class GraphVizDrawer:
    def __init__(self) -> None:
        pass

    def invoke(self) -> str:
        pass

def get_drawer(drawer: str) -> Callable:
    pass

class YamlWorkflowVisualizer(Tool):
    def print_usage(self) -> str:
        return """
            'Draw an image of a workflow in yaml format.',
            'Inputs: {
                "workflow": str (the workflow in yaml format), 
                "llm": str (the LLM used to generate the prompt, default is "deepseek-chat"), 
                "drawer": str (the model used to generate the image, default is "graphviz"), 
            }',
            "Outputs: str (the path of the generated png figure)."
        """

    def run(self,
        workflow: str,
        llm: str="deepseek-chat",
        drawer: str="graphviz",
    ) -> Union[List[str], List[str]]:

        return [os.path.abspath(img_file)], [os.path.abspath(img_file)]

class CodeWorkflowVisualizer(Tool):
    def __init__(self) -> None:
        pass

    def print_usage(self) -> str:
        return "\n".join([
            'Visualize a workflow implemented with Python code.',
            'Inputs: {"workflow": the workflow (should be part of the workflow), "rotate": whether to rotate the molecule}',
            "Outputs: A figure."
        ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="visualize_molecule")
    parser.add_argument("--molecule", type=str, default=None)
    parser.add_argument("--molecule_config", type=str, default=None)
    parser.add_argument("--protein", type=str, default=None)
    parser.add_argument("--protein_config", type=str, default=None)
    parser.add_argument("--pocket", type=str, default=None)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--save_output_filename", type=str, default=None)
    parser.add_argument("--color", type=str, default=None, choices=["grey", "spectrum"])
    
    args = parser.parse_args()
    if args.task == "visualize_molecule":
        img_file = MoleculeVisualizer().run(
            create_tool_input("molecule", args.molecule),
            config=args.molecule_config,
            img_file=args.output_file,
        )[0]
    elif args.task == "visualize_protein":
        protein_vis = ProteinVisualizer()
        config = args.protein_config
        if args.color and config:
            # Load config first, then override color
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cfg_path = os.path.join(project_root, "configs", "visualization", "protein", f"{config}.yaml")
            config = merge_config(
                Config(cfg_path),
                Config(os.path.join(project_root, "configs", "visualization", "global_config.yaml"))
            )
            config.protein.color = args.color
        img_file = protein_vis.run(
            create_tool_input("protein", args.protein),
            config=config,
            img_file=args.output_file,
            rotate=args.rotate
        )[0]
    elif args.task == "visualize_complex":
        img_file = ComplexVisualizer().run(
            create_tool_input("molecule", args.molecule),
            create_tool_input("protein", args.protein),
            molecule_config=args.molecule_config,
            protein_config=args.protein_config,
            img_file=args.output_file,
            rotate=args.rotate
        )[0]
    elif args.task == "visualize_protein_pocket":
        img_file = ProteinPocketVisualizer().run(
            create_tool_input("protein", args.protein),
            create_tool_input("pocket", args.pocket),
            img_file=args.output_file,
        )[0]
    print(img_file)
    if args.save_output_filename is not None:
        with open(args.save_output_filename, "w") as f:
            f.write(img_file[0])