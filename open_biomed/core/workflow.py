from typing import Any, Dict, TextIO, List, Tuple, Optional, TypedDict

import argparse
import asyncio
import copy
import json
import numpy as np
import os
import uuid
import yaml
import re
import subprocess
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
work_dir = os.path.abspath(__file__).replace("open_biomed/core/workflow.py", "")

from open_biomed.tools.tool_registry import TOOLS
from open_biomed.tools.visualization_tools import Visualizer
from open_biomed.utils.config import Config
from open_biomed.data import Molecule, Protein, Text
from open_biomed.utils.misc import wrap_outputs, wrap_and_select_outputs, create_tool_input

def parse_frontend(json_string: str, output_floder: str = "tmp/workflow") -> str:
    
    param_mapping = {
        "molecule_name_request": {"query": "accession"},
        "molecule_structure_request": {"query": "accession"},
        "protein_uniprot_request": {"query": "accession"},
        "protein_pdb_request": {"query": "accession"},
        "pubchemid_search": {"query": "accession"},
        "molecule_property_prediction": {"dataset": "task"}
    }
    def get_createdata_node(node):
        node_dict = {}
        id = node["id"]
        node_dict["value"] = {}
        description = node["data"]["node"]["description"]
        data_context = node["data"]["node"]["template"]
        keys = list(data_context)
        pattern = re.compile(r'^field_\d+_key$')  # get param
        filtered_keys = [key for key in keys if pattern.match(key)]
        for key in filtered_keys:
            node_dict["value"].update(data_context[key]["value"])
        return node_dict

    # just get id and description
    def get_tool_node(node):
        node_dict = {}
        node_dict["value"] = {}
        id = node["id"]
        description = node["data"]["node"]["description"]
        node_dict["description"] = [description]
        data_context = node["data"]["node"]["template"]
        keys = list(data_context)
        for key in keys:
            if key in ["config", "molecule", "protein", "pocket", "text", "dataset", "query", "mutation", "indices", "threshold"] and data_context[key]["value"]:
                node_dict["value"].update({key: data_context[key]["value"]})
        return node_dict

    def remove_duplicate_path(nodes):
        # node handled
        seen = set()
        new_nodes = []
        for head, tail in nodes:
            if (head, tail) not in seen:
                new_nodes.append([head, tail])
                seen.add((head, tail))
            else:
                print(f"The duplicate path has been removed: {head} -> {tail}")

        return new_nodes

    def check_strings(str_list, nested_list):
        for sublist in nested_list:
            for str in str_list:
                if str in sublist:
                    return False
        return True

    def merge_nodes(nodes):
        keywords = ["MergeDataComponent", "ParseData"]
        node_list_copy = copy.deepcopy(nodes)
        new_nodes_list = []
        for i, (head, tail) in enumerate(node_list_copy):
            if any(keyword in head for keyword in keywords):
                for j, (next_head, next_tail) in enumerate(node_list_copy):
                    if i != j and head == next_tail:
                        # merge path
                        new_nodes_list.append((next_head, tail))
            if any(keyword in tail for keyword in keywords):
                for j, (next_head, next_tail) in enumerate(node_list_copy):
                    if i != j and tail == next_head:
                        # merge path
                        new_nodes_list.append((head, next_tail))

        new_nodes_list = [list(i) for i in set(new_nodes_list)]
        
        return nodes, new_nodes_list

    try:
        node_edge_data = json.loads(json_string)
        print("JSON successfully loaded！")
    except json.JSONDecodeError as e:
        print("JSON loading error", e)
    try:
        tool_node = {}
        createdata_node = {}
        nodes_data = node_edge_data["data"]["nodes"] if "data" in node_edge_data else node_edge_data["nodes"]
        for node in nodes_data:
            if "ChatInput" in node["id"] or "ChatOutput" in node["id"] or "ParseData" in node["id"]:
                continue
            if "PharmolixCreateData" in node["id"]:
                data = get_createdata_node(node)
                createdata_node[node["id"]] = data
            else:
                data = get_tool_node(node)
                tool_node[node["id"]] = data

        edge_list = []
        edges_data = node_edge_data["data"]["edges"] if "data" in node_edge_data else node_edge_data["edges"]
        for node in edges_data:
            source = node["source"]
            target = node["target"]
            if "PharmolixCreateData" in source:
                tool_node[target]["value"] = createdata_node[source]["value"]
                edge_list.append([source, target])
            else:
                edge_list.append([source, target])

        path_nodes = copy.deepcopy(edge_list)
        path_nodes, new_path_nodes = merge_nodes(path_nodes)

        # get value for tools after merge
        for node in new_path_nodes:
            source = node[0]
            target = node[1]
            if "PharmolixCreateData" in source:
                tool_node[target]["value"] = createdata_node[source]["value"]

        merge_nodes_list = path_nodes + new_path_nodes

        # remove paths containing path_keywords
        path_keywords = ["MergeDataComponent", "ParseData", "ChatOutput", "ChatInput", "Image Output", "PharmolixCreateData", "Image output"]
        merge_nodes_list_copy = copy.deepcopy(merge_nodes_list)
        for index in range(len(merge_nodes_list_copy) - 1, -1, -1):
            # print(index)
            value = merge_nodes_list_copy[index]
            if any(keyword in value[0] for keyword in path_keywords) or any(keyword in value[1] for keyword in path_keywords):
                merge_nodes_list.pop(index)

        # get tool nodes info
        tool_node_filted = {}
        key_list = []
        for key, value in tool_node.items():
            if not any(keyword in key for keyword in path_keywords):
                tool_node_filted[key] = value
                key_list.append(key)

        # get the list of node in paths
        path_nodes_list = []
        for i in merge_nodes_list:
            path_nodes_list.append(i[0])
            path_nodes_list.append(i[1])
        path_nodes_list = list(set(path_nodes_list))

        assert len(path_nodes_list) == len(key_list), "please check the frontend json file"

        yaml_dict = {}
        yaml_dict["tools"] = []
        yaml_dict["edges"] = []
        for node in path_nodes_list:
            node_info = tool_node_filted[node]
            if 'value' in node_info.keys() and node_info["value"]:
                yaml_dict["tools"].append({"name": node.split("-")[0].lower(), "inputs": node_info["value"]})
            else:
                yaml_dict["tools"].append({"name": node.split("-")[0].lower()})

        for path in merge_nodes_list:
            yaml_dict["edges"].append({"start": path_nodes_list.index(path[0]), "end": path_nodes_list.index(path[1])})

        
        # param_mapping
        # TODO: hard code
        for node in yaml_dict["tools"]:
            if "inputs" in node and "model" in list(node["inputs"].keys()):
                node["inputs"].pop("model")
            if node["name"] in param_mapping:
                print(node["name"])
                for key in list(node["inputs"].keys()):
                    if key in param_mapping[node["name"]]:
                        new_key = param_mapping[node["name"]][key]
                        node["inputs"][new_key] = node["inputs"].pop(key)

        
        if not os.path.exists(output_floder):
            os.makedirs(output_floder)

        #file_name_with_extension = os.path.basename(file_path)
        #file_name_without_extension = os.path.splitext(file_name_with_extension)[0]
        uid = str(uuid.uuid4())
        output_path = os.path.join(output_floder, f"{uid}.yaml")

        with open(output_path, "w", encoding="utf-8") as file:
            yaml.dump(yaml_dict, file, allow_unicode=True, sort_keys=False)

        # node_edge_data saved as json file
        with open(output_path.replace(".yaml", ".json"), "w", encoding="utf-8") as file:
            json.dump(node_edge_data, file, ensure_ascii=False, indent=4)

        print(f"yaml file has been saved to {output_path}")
        return output_path
    except Exception as e:
        print("Error:", e)
        print("Please ensure that your workflow can run successfully before submitting")
        uid = str(uuid.uuid4())
        output_path = os.path.join(output_floder, f"{uid}.json")
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(node_edge_data, file, ensure_ascii=False, indent=4)
        print(output_path)
        return "Please ensure that your workflow can run successfully before submitting"


def get_str_from_dict(input: Dict[str, Any]) -> str:
    input_str = copy.deepcopy(input)
    for key in input_str:
        input_str[key] = str(input_str[key])
    return json.dumps(input_str)

class WorkflowNode():
    def __init__(self,
        name: str,
        executable: Any,
        inputs: Dict[str, Any]={},
        num_repeats: int=1,
    ) -> None:
        self.name = name
        self.executable = executable
        self.inputs = {}
        for key, value in inputs.items():
            self.inputs[key] = create_tool_input(key, value)
        self.orig_inputs = copy.deepcopy(self.inputs)
        self.num_repeats = num_repeats
        self.outputs = None

    def pretty_print(self, step_id: int) -> str:
        completed = self.outputs is not None
        message = f"Node {step_id + 1}. "
        message += "[✓] " if completed else "[ ] "
        message += self.name.upper() + "\n"
        if self.name == "code_execution":
            message += f"Execute the following code for {self.num_repeats} times: \n```python\n" + self.inputs["code"] + "\n```"
        else:
            message += self.executable.print_usage().split("\n")[0]
            if self.num_repeats > 1:
                message += f" (repeat {self.num_repeats} times)"
            else:
                message += "\n"
            message += "Filled inputs:\n"
            for key, value in self.inputs.items():
                message += f"{key}: {value}\n"
        if completed:
            message += "Outputs:\n"
            for i in range(len(self.outputs[1])):
                message += f"{self.outputs[1][i]}\n"
        message += "\n"
        return message

class WorkflowV0():
    def __init__(self, config: Config) -> None:
        # create a list of tools as nodes and a DAG as edges for the workflow to run
        self.nodes = []
        self.edges = []
        for tool in config.tools:
            self.nodes.append(WorkflowNode(
                name=tool["name"],
                executable=TOOLS[tool["name"]],
                inputs=tool.get("inputs", {})
            ))
        for edge in config.edges:
            self.edges.append((edge["start"], edge["end"], edge.get("name_mapping", None)))

    @classmethod
    def from_forntend_export(cls, file: str) -> None:
        yml_file = parse_frontend(file)
        return cls(config=Config(config_file=yml_file))

    async def run(self, num_repeats=10, context: TextIO=sys.stdout, tool_outputs: TextIO=sys.stdout) -> Any:
        context.write("Now we have a workflow with the following tools: \n")
        for i, node in enumerate(self.nodes):
            context.write(f"Tool No.{i + 1}: {node.executable.print_usage()}\n\n\n")
        for repeat in range(num_repeats):
            try:
                context.write(f"Repeating workflow execution No.{repeat + 1}:\n")
                edge_list = [[] for i in range(len(self.nodes))]
                in_deg = [0 for i in range(len(self.nodes))]
                for edge in self.edges:
                    edge_list[edge[0]].append((edge[1], edge[2]))
                    in_deg[edge[1]] += 1
                queue = []
                for i in range(len(self.nodes)):
                    self.nodes[i].inputs = copy.deepcopy(self.nodes[i].orig_inputs)
                    if in_deg[i] == 0:
                        queue.append(i)
                while len(queue) > 0:
                    u = queue.pop(0)
                    context.write(f"Next we execute Tool No.{u + 1}.\n The inputs of this tool are: {get_str_from_dict(self.nodes[u].inputs)}\n")
                    if getattr(self.nodes[u].executable, "requires_async", False):
                        outputs = await asyncio.create_task(self.nodes[u].executable.run(**self.nodes[u].inputs))
                    elif isinstance(self.nodes[u].executable, Visualizer):
                        # Execute visualization in a subprocess to avoid internal errors of PyMol when an InferencePipeline object is created
                        vis_process = [
                            "python3", "./open_biomed/core/visualize.py", 
                            "--task", self.nodes[u].name,
                            "--save_output_filename", "./tmp/visualization_file.txt",
                        ]
                        for key, value in self.nodes[u].inputs.items():
                            if key in ["molecule", "protein", "pocket"]:
                                vis_process.append(f"--{key}")
                                if key == "molecule":
                                    vis_process.append(value.save_sdf())
                                elif key == "protein":
                                    vis_process.append(value.save_pdb())
                                elif key == "pocket":
                                    vis_process.append(value.save_binary())
                        subprocess.Popen(vis_process).communicate()
                        outputs = open("./tmp/visualization_file.txt", "r").read()
                        outputs = [outputs], [outputs]
                    else:
                        outputs = self.nodes[u].executable.run(**self.nodes[u].inputs)

                    tool_name = self.nodes[u].executable.print_usage().split(".\n")[0].lower().split()
                    dir_name, file_name = os.path.dirname(outputs[1][0]), os.path.basename(outputs[1][0])
                    file_name = f"tool_{u+1}_"+ "_".join(tool_name) + "_" + file_name
                    if isinstance(outputs[0][0], Molecule) and hasattr(outputs[0][0], "conformer"):
                        file_name = file_name.replace("pkl", "sdf")
                        file_path = os.path.join(dir_name, file_name)
                        outputs[0][0].save_sdf(file_path)
                        tool_outputs.write(file_path+"\n")
                    elif isinstance(outputs[0][0], Protein) and hasattr(outputs[0][0], "conformer"):
                        file_name = file_name.replace("pkl", "pdb")
                        file_path = os.path.join(dir_name, file_name)
                        outputs[0][0].save_pdb(file_path)
                        tool_outputs.write(file_path+"\n")
                    elif outputs[1][0][-4:] in [".png"] and os.path.exists(outputs[1][0]):
                        file_path = os.path.join(dir_name, file_name)
                        os.rename(outputs[1][0], file_path)
                        tool_outputs.write(file_path+"\n")
                    
                    outputs = wrap_and_select_outputs(outputs, context)
                    context.write(f"The outputs of this tool are: {get_str_from_dict(outputs)}\n\n\n")
                    for out_node, out_name_mapping in edge_list[u]:
                        in_deg[out_node] -= 1
                        if in_deg[out_node] == 0:
                            queue.append(out_node)
                        for key, value in outputs.items():
                            if out_name_mapping is not None and key in out_name_mapping:
                                key = out_name_mapping[key]
                            context.write(f'Tool No.{u + 1} passes its "{key}" output to Tool No.{out_node + 1}\n')
                            self.nodes[out_node].inputs[key] = copy.deepcopy(value)
            except Exception as e:
                print(e)
                context.write("The workflow execution failed")

class WorkflowState(TypedDict):
    exec_queue: List[int]
    in_deg: List[int]
    node_list: List[WorkflowNode]
    edge_list: List[Tuple[int, int, Dict[str, Any]]]
    messages: List[str]

class Workflow:
    def __init__(self, config: Config) -> None:
        self.metadata = config.metadata
        self.description = config.metadata.description + "\nExpected inputs: "
        for missing_input in config.metadata.inputs:
            self.description += f"The '{missing_input[1]}' field of for Step '{missing_input[0]}' ({missing_input[2]})\n"
        self.nodes = []
        self.edges = []
        for tool in config.tools:
            if tool["name"] == "code_execution":
                self.nodes.append(WorkflowNode(
                    name="code_execution",
                    executable=exec,
                    inputs={"code": tool.get("code", {})},
                    num_repeats=tool.get("num_repeats", 1)
                ))
            else:
                self.nodes.append(WorkflowNode(
                    name=tool["name"],
                    executable=TOOLS[tool["name"]],
                    inputs=tool.get("inputs", {}),
                    num_repeats=tool.get("num_repeats", 1)
                ))
        out_deg = [0 for i in range(len(self.nodes))]
        for edge in config.edges:
            out_deg[edge["start"]] += 1
            self.edges.append((edge["start"], edge["end"], edge.get("name_mapping", None)))
        self.output_nodes = [i for i in range(len(self.nodes)) if out_deg[i] == 0]
        if hasattr(self.metadata, "outputs"):
            for output in self.metadata.outputs:
                assert output[0] in self.output_nodes, f"The output {output[0]} is not a valid output node"
            self.output_nodes = [output[0] for output in self.metadata.outputs]
        else:
            self.metadata.outputs = []
            for i in self.output_nodes:
                if self.nodes[i].name == "code_execution":
                    self.metadata.outputs.append([i, "Any: The code execution result"])
                else:
                    self.metadata.outputs.append([i, self.nodes[i].executable.print_usage().split("\n")[2]])
        self.output_nodes.sort()

    def step(self, state: WorkflowState) -> WorkflowState:
        state["exec_queue"].sort()
        node_idx = state["exec_queue"].pop(0)
        node = state["node_list"][node_idx]
        node.outputs = ([], [])
        print(node.inputs)
        for i in range(node.num_repeats):
            if getattr(node.executable, "requires_async", False):
                outputs = asyncio.run(node.executable.run(**node.inputs))
            elif node.name == "code_execution":
                node_inputs, node_outputs = [], []
                for cur_node in state["node_list"]:
                    if cur_node.outputs is not None:
                        node_outputs.append(copy.deepcopy(cur_node.outputs[0]))
                    else:
                        node_outputs.append(None)
                    node_inputs.append(copy.deepcopy(cur_node.inputs))
                exec(node.inputs["code"], {"step_inputs": node_inputs, "step_outputs": node_outputs})
                outputs = [node_outputs[node_idx]], ["Code execution completed"]
            else:
                outputs = node.executable.run(**node.inputs)
            node.outputs[0].extend(outputs[0])
            node.outputs[1].extend(outputs[1])
        for e in self.edges:
            if e[0] == node_idx:
                state["in_deg"][e[1]] -= 1
                if node.name == "code_execution":
                    for key, value in node_inputs[e[1]].items():
                        state["node_list"][e[1]].inputs[key] = copy.deepcopy(value)
                    print(node.outputs)
                    wrapped_outputs = node.outputs[0][0]
                else:
                    wrapped_outputs = wrap_outputs(node.outputs[0])
                for key, value in wrapped_outputs.items():
                    if e[2] is not None and key in e[2]:
                        key = e[2][key]
                    state["node_list"][e[1]].inputs[key] = copy.deepcopy(value)
                if state["in_deg"][e[1]] == 0:
                    state["exec_queue"].append(e[1])
        state["messages"].append(node.pretty_print(node_idx))

    def determine_next_step(self, state: WorkflowState) -> bool:
        if len(state["exec_queue"]) == 0 or self.deamon is not None and self.deamon.should_interrupt(state):
            return False
        else:
            return True

    def _initial_state(self, inputs: Tuple[int, str, Any]) -> WorkflowState:
        in_deg = [0 for i in range(len(self.nodes))]
        for edge in self.edges:
            in_deg[edge[1]] += 1
        node_list = self.nodes
        for node in node_list:
            node.inputs = node.orig_inputs
        for elem in inputs:
            node_list[elem[0]].inputs[elem[1]] = elem[2]
        return {
            "exec_queue": [i for i in range(len(self.nodes)) if in_deg[i] == 0],
            "in_deg": in_deg,
            "node_list": node_list,
            "edge_list": self.edges,
            "messages": [self.description],
        }

    def run(self, inputs: Tuple[int, str, Any], num_repeats: int=1, deamon: Optional[Any]=None) -> Tuple[List[Any], List[str]]:
        self.deamon = deamon
        results, messages = [], []
        for i in range(num_repeats):
            state = self._initial_state(inputs)
            while self.determine_next_step(state):
                self.step(state)
            results.append([])
            for node_idx in self.output_nodes:
                results[-1].append(copy.deepcopy(state["node_list"][node_idx].outputs[0]))
            messages.append(copy.deepcopy(state["messages"]))
        return results, messages

WORKFLOWS = {}
for file in os.listdir(os.path.join(work_dir, "memory/workflows")):
    workflow_name = file.split(".")[0]
    workflow_config = Config(config_file=os.path.join(work_dir, "memory/workflows", file))
    WORKFLOWS[workflow_name] = Workflow(workflow_config)

if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", type=str, default="stable_drug_design")
    parser.add_argument("--num_repeats", type=int, default=1)
    args = parser.parse_args()

    config = Config(config_file=f"./configs/workflow/{args.workflow}.yaml")
    workflow = Workflow(config)
    asyncio.run(workflow.run(num_repeats=args.num_repeats, context=open("./logs/workflow_outputs.txt", "w"), tool_outputs=open("./logs/workflow_tool_outputs.txt", "w")))
    # workflow.run(num_repeats=1)
    file_path = "configs/workflow/yuandong-0404.json"
    with open(file_path, "r") as f:
        json_data = json.load(f)
    json_string = json.dumps(json_data, ensure_ascii=False, indent=4)  # if 
    fronted_file = parse_frontend(json_string)
    """
    """
    file_path = "memory/workflows/pdb_query.yaml"
    config = Config(config_file=file_path)
    workflow = Workflow(config)
    results, messages = workflow.run([(0, "accession", "4xli"), (1, "accession", "4xli")])
    print(len(results))
    print(results[0][0], results[0][1])
    with open("./tmp/pdb_query_messages.txt", "w") as f:
        for msg in messages[0]:
            f.write(msg + "\n")
    """
    file_path = "memory/generated_workflows/visualize_pdb.yaml"
    config = Config(config_file=file_path)
    workflow = Workflow(config)
    results, messages = workflow.run([(0, "pdb_file", "./tmp/4xli.pdb"), (1, "chain_id", "A")])
    print(results[0][0])
    for msg in messages[0]:
        print(msg)