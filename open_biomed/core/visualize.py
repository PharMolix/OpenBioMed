from typing import List, Optional, Union
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
from datetime import datetime
import shutil
from rdkit.Chem import Draw, rdDepictor

from open_biomed.core.tool import Tool
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
            if hasattr(config.protein, 'show'):
                for elem in config.protein.show:
                    cmd.show(elem, "protein")
            for elem in config.protein.__dict__.keys():
                if elem not in ["color", "show", "cnc"]:
                    cmd.set(elem, config.protein.__dict__[elem], "protein")
            cmd.color(getattr(config.protein, "color", "grey"), "protein")
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
            'Inputs: {"molecule": a small molecule, "rotate": whether to rotate the molecule}',
            "Outputs: A figure."
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
            'Visualize protein.',
            'Inputs: {"protein": a protein (3D structure is required), "rotate": whether to rotate the molecule}',
            "Outputs: A figure."
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
            'Inputs: {"molecule": the ligand, "protein": the protein receptor, "rotate": whether to rotate the molecule}',
            "Outputs: A figure."
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="visualize_molecule")
    parser.add_argument("--molecule", type=str, default=None)
    parser.add_argument("--molecule_config", type=str, default=None)
    parser.add_argument("--protein", type=str, default=None)
    parser.add_argument("--protein_config", type=str, default=None)
    parser.add_argument("--pocket", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--save_output_filename", type=str, default=None)
    
    args = parser.parse_args()
    if args.task == "visualize_molecule":
        img_file = MoleculeVisualizer().run(
            create_tool_input("molecule", args.molecule),
            config=args.molecule_config,
            img_file=args.output_file,
        )[0]
    elif args.task == "visualize_protein":
        img_file = ProteinVisualizer().run(
            create_tool_input("protein", args.protein),
            config=args.protein_config,
            img_file=args.output_file,
        )[0]
    elif args.task == "visualize_complex":
        img_file = ComplexVisualizer().run(
            create_tool_input("molecule", args.molecule),
            create_tool_input("protein", args.protein),
            molecule_config=args.molecule_config,
            protein_config=args.protein_config,
            img_file=args.output_file,
            rotate=False
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