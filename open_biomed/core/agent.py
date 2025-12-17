from abc import ABC, abstractmethod
import aiodocker
import asyncio
import copy
from dotenv import load_dotenv
import io
from typing import Annotated, Sequence, TypedDict
import os
import pickle
import re
import sqlite3
import subprocess
import sys
import traceback
import uuid
import yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
work_dir = os.path.abspath(__file__).replace("/open_biomed/core/agent.py", "")

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.base import get_msg_title_repr
from langchain_core.utils.interactive_env import is_interactive_env
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from open_biomed.core.llm_provider import get_llm
from open_biomed.core.workflow import WORKFLOWS
from open_biomed.tools.tool_registry import TOOLS
from open_biomed.utils.config import Config

load_dotenv(".env")
OPENBIOMED_SERVICE_URL = os.getenv("OPENBIOMED_SERVICE_URL")

_patched = False
_persistent_namespace = {}

def get_thread_id(agent_name: str) -> int:
    thread_id = str(uuid.uuid4())
    os.makedirs(f"{work_dir}/tmp/{agent_name}-{thread_id}", exist_ok=True)
    return thread_id

def dump_namespace(namespace: dict, file_path: str):
    new_namespace = {}
    for key, value in namespace.items():
        if key != "__builtins__":
            try:
                new_namespace[key] = pickle.dumps(value)
            except Exception as e:
                # print(f"Error dumping {key}: {value}")
                pass
    pickle.dump(new_namespace, open(file_path, "wb"))

def load_namespace(file_path: str) -> dict:
    new_namespace = {}
    with open(file_path, "rb") as f:
        namespace = pickle.load(f)
    for key, value in namespace.items():
        if key != "__builtins__":
            new_namespace[key] = pickle.loads(value)
    return new_namespace

def _apply_saving_patches(agent_name: str, thread_id: str, captured_results: list[tuple[str, str]]):
    import matplotlib.pyplot as plt
    from open_biomed.data import Molecule, Protein
    from open_biomed.tools.visualization_tools import VisualizerWrapper
    global _patched
    if _patched:
        return
    orig_show = plt.show
    orig_savefig = plt.savefig
    orig_savemol = Molecule.save_sdf
    orig_saveprotein = Protein.save_pdb
    orig_visualize = VisualizerWrapper.run
    work_dir = os.path.abspath(__file__).replace("/open_biomed/core/agent.py", "")

    def custom_show(*args, **kwargs):
        orig_show(*args, **kwargs)
        plt.savefig(f"matplotlib_plot.png")

    def custom_savefig(*args, **kwargs):
        if len(args) > 0:
            filename = args[0]
            args.pop(0)
        else:
            filename = kwargs["fname"]
            kwargs.pop("fname")
        filename = filename.split("/")[-1]
        captured_results.append(("figure", f"{work_dir}/tmp/{agent_name}-{thread_id}/{filename}"))
        orig_savefig(filename, *args, **kwargs)

    def custom_savemol(*args, **kwargs):
        if len(args) > 1:
            filename = args[1].split("/")[-1]
            filename = f"{work_dir}/tmp/{agent_name}-{thread_id}/{filename}"
            args = (args[0], filename) + args[2:]
        elif "file" in kwargs:
            filename = kwargs["file"].split("/")[-1]
            filename = f"{work_dir}/tmp/{agent_name}-{thread_id}/{filename}"
            kwargs["file"] = filename
        else:
            args[0]._add_name()
            filename = f"{work_dir}/tmp/{agent_name}-{thread_id}/{args[0].name}.sdf"
            kwargs = {**kwargs, "file": filename}
        captured_results.append(("molecule", filename))
        return orig_savemol(*args, **kwargs)

    def custom_saveprotein(*args, **kwargs):
        if len(args) > 1:
            filename = args[1].split("/")[-1]
            filename = f"{work_dir}/tmp/{agent_name}-{thread_id}/{filename}"
            args = (args[0], filename) + args[2:]
        elif "file" in kwargs:
            filename = kwargs["file"].split("/")[-1]
            filename = f"{work_dir}/tmp/{agent_name}-{thread_id}/{filename}"
            kwargs = {**kwargs, "file": filename}
        else:
            args[0]._add_name()
            filename = f"{work_dir}/tmp/{agent_name}-{thread_id}/{args[0].name}.pdb"
            kwargs = {**kwargs, "file": filename}
        captured_results.append(("protein", filename))
        return orig_saveprotein(*args, **kwargs)

    def custom_visualize(*args, **kwargs):
        outputs = orig_visualize(*args, **kwargs)
        new_outputs, new_messages = [], []
        for output in outputs[0]:
            filename = output.split("/")[-1]
            filename = f"{work_dir}/tmp/{agent_name}-{thread_id}/{filename}"
            captured_results.append(("visualization", filename))
            os.system(f"mv {output} {filename}")
            new_outputs.append(filename)
            new_messages.append(f"The generated figure is saved at {filename}")
        return new_outputs, new_messages

    plt.show = custom_show
    plt.savefig = custom_savefig
    Molecule.save_sdf = custom_savemol
    Protein.save_pdb = custom_saveprotein
    VisualizerWrapper.run = custom_visualize
    _patched = True

async def exec_cmd(code: str, timeout: int=10) -> tuple[int, str, str]:
    """Execute a command in a shell and capture output.

    Args:
        code: Command to execute as string
        timeout: Maximum time in seconds to wait for command completion

    Returns:
        tuple containing:
            - Exit code from command execution
            - stdout output as string
            - stderr output as string
    """
    process = await asyncio.create_subprocess_shell(
        code,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        print(f"Command timed out after {timeout} seconds. Terminating process...")
        
        # Terminate the process gently first
        try:
            process.terminate()
            # Wait a short while for the process to acknowledge termination
            await asyncio.wait_for(process.wait(), timeout=5.0) 
        except asyncio.TimeoutError:
            # If termination fails, kill the process forcefully
            print("Termination failed, killing process.")
            process.kill()
            await process.wait() # Wait for the kill to complete
            
        return 1, "", f"Command execution timed out after {timeout} seconds"
    
    # Read output only after the process has finished (process.wait() has completed)
    stdout, stderr = await process.communicate()

    # Decode the byte output to string
    return [process.returncode, stdout.decode().strip(), stderr.decode().strip()]

async def exec_cmd_docker(command: str, container: aiodocker.Docker, timeout: int=10) -> tuple[int, str, str]:
    """Execute a command in a Docker container and capture output.

    Args:
        container: Docker container instance to execute command in
        exec_command: Command to execute as list of strings
        timeout: Maximum time in seconds to wait for command completion

    Returns:
        tuple containing:
            - Exit code from command execution
            - stdout output as string
            - stderr output as string

    Raises:
        TimeoutError: If command execution exceeds timeout period
    """
    pass

async def exec_python(code: str, timeout: int=10, exec_namespace: dict={}) -> tuple[int, str, str]:
    """Execute a Python code and capture output.
    """
    def custom_exec_python(code: str):
        try:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            exec(code, exec_namespace)
            stdout = sys.stdout.getvalue()
            stderr = sys.stderr.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            return [0, stdout, stderr]
        except Exception as e:
            traceback.print_exc()
            stdout = sys.stdout.getvalue()
            stderr = sys.stderr.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            return [1, stdout, stderr]
    process = asyncio.to_thread(custom_exec_python, code)
    try:
        result = await asyncio.wait_for(process, timeout=timeout)
    except asyncio.TimeoutError:
        return [1, "", f"Python code execution timed out after {timeout} seconds"]
    return result

async def exec_python_docker(code: str, container: aiodocker.Docker, timeout: int=10) -> tuple[int, str, str]:
    """Execute a Python code in a Docker container and capture output.
    """
    pass

def pretty_print(message: BaseMessage) -> None:
    title = message.type.title() if not "<observation>" in message.content else "Tool"
    content = get_msg_title_repr(title + " Message", bold=is_interactive_env())
    if message.name is not None:
        content += f"\nName: {message.name}"
    content += f"\n\n{message.content}"
    print(content)

class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    num_steps: int
    chat_failure_count: int
    code_failure_count: int
    next_step: str
    exec_namespace: dict

def print_state(state: AgentState) -> None:
    print("Num steps: " + str(state["num_steps"]))
    print("Chat failure count: " + str(state["chat_failure_count"]))
    print("Code failure count: " + str(state["code_failure_count"]))
    print("Next step: " + str(state["next_step"]))
    for message in state["messages"]:
        pretty_print(message)
    print("===============================")

class Agent(ABC):
    def __init__(
        self,
        agent_cfg: Config,
    ):
        self.timeout = agent_cfg.timeout

    @abstractmethod
    def _init_tools(self):
        pass

    @abstractmethod
    def _init_system_prompt(self):
        pass

    @abstractmethod
    def run(self, user_prompt: str):
        pass

class PlannerExecutor(Agent):
    # PlannerExecutor is responsible for:
    # 1. Designing the plan
    # 2. Executing the plan
    # 3. Writing the report
    def __init__(self,
        agent_cfg: Config,
    ):
        super(PlannerExecutor, self).__init__(agent_cfg)
        self.plan_style = getattr(agent_cfg, "plan_style", "checklist")
        if self.plan_style not in ["checklist", "step-by-step"]:
            raise ValueError(f"Invalid plan style: {self.plan_style}")
        self.critic = getattr(agent_cfg, "critic", False)
        self.tool_retriever = getattr(agent_cfg, "tool_retriever", None)
        self.tool_call = getattr(agent_cfg, "tool_call", "custom")
        if self.tool_call not in ["custom", "http_request"]:
            raise ValueError(f"Invalid tool call mode: {self.tool_call}")
        self.chat_tolerance = getattr(agent_cfg, "chat_tolerance", 3)
        self.code_tolerance = getattr(agent_cfg, "code_tolerance", 3)
        self.use_docker = getattr(agent_cfg, "use_docker", False)
        if self.use_docker:
            self.docker_container_id = getattr(agent_cfg, "docker_container_id", "youngking0727/openbiomed_server")
        self.checkpointing = getattr(agent_cfg, "checkpointing", False)
        self.checkpointing_db_path = getattr(agent_cfg, "checkpointing_db_path", None)
        self.memory = getattr(agent_cfg, "memory", {})

        self.llm = get_llm(agent_cfg.llm, stop_sequences=["</execute>", "</report>"])
        self.llm_model_name = agent_cfg.llm
        self._init_tools()
        self._init_memory()
        self._init_system_prompt()
        self._setup_execution_environment()
        self._init_agent_workflow()

    def _init_tools(self):
        self.tools = TOOLS

    def _init_memory(self):
        if getattr(self.memory, "workflow", False):
            self.workflows = WORKFLOWS

    def _init_system_prompt(self):
        if self.plan_style == "checklist":
            plan_demo = """
Format your plan as a checklist with empty checkboxes like this:

1. [ ] Action 1
2. [ ] Action 2
3. [ ] Action 3
...
"""
            plan_success_demo = """
After completing each step, update the checklist by replacing the empty checkbox with a checkmark, and add a brief summary of the action results:

1. [✓] Action 1
<summary>
A brief summary of the action results.
</summary>
2. [ ] Action 2
3. [ ] Action 3
...
"""
            plan_failed_demo = """
If a step fails or needs modification, mark it with an X, explain why, and try to update the action to a new one:

1. [✓] Action 1
<summary>
A brief summary of the action results.
</summary>
2. [✗] Action 2
<summary>
A brief summary of why the action failed.
</summary>
3. [ ] Modified Action 2
4. [ ] Action 3
...
"""
        elif self.plan_style == "step-by-step":
            plan_demo = """
Format your plan as a markdown table like this:

| Step | Action   | Status  | Results |
|------|----------|---------|---------|
| 1    | Action 1 | Pending |         |
| 2    | Action 2 | Pending |         |
| 3    | Action 3 | Pending |         |
| ...  | ...      | ...     | ...     |
"""
            plan_success_demo = """
After completing each step, update the table by replacing the Pending status with Completed, and add the results of the action:

| Step | Action   | Status  | Results |
|------|----------|---------|---------|
| 1    | Action 1 | Completed | A brief summary of the action results. |
| 2    | Action 2 | Pending |         |
| 3    | Action 3 | Pending |         |
| ...  | ...      | ...     | ...     |
"""
            plan_failed_demo = """
If a step fails or needs modification, mark it with an X, explain why, and try to update the action to a new one:

| Step | Action   | Status  | Results |
|------|----------|---------|---------|
| 1    | Action 1 | Completed | A brief summary of the action results. |
| 2    | Action 2 | Failed | A brief summary of why the action failed. |
| 3    | Modified Action 2 | Pending |         |
| 4    | Action 3 | Pending |         |
| ...  | ...      | ...     | ...     |
"""
        self.prompt = f"""
You are a professionoal and helpful biomedical assistant. Your task is to solve complicated biomedical research problems.

To achieve this, you will be working with an interactive coding environment with a variety of functions and data sources to help you throughout the whole process. 

Given a task, you should first design a plan for the team to achieve the task. The plan should be a list of steps, each step should be a description of the action to be taken by yourself or another agent.

{plan_demo}

Then, you should execute the plan STEP BY STEP.

{plan_success_demo}

{plan_failed_demo}

Always show the updated plan after each step so the user can track progress.

At each step, you should first show your reasoning based on the conversation history. During execution, you can write codes and interact with a programming environment. Your code should be enclosed in "<execute>...</execute>" tags. The execution result will be shown within "<observation>...</observation>" tags. The programming environment supports the following languages:
- For Python code (default): <execute> print("Hello, world!") </execute>
- For bash scripts: <execute> #!BASH echo "Hello, world!" </execute>
- For CLI software, use bash scripts.
You are allowed to interact with the programming environment multiple times in each step. So you can decompose your code into multiple parts.
Do not write one large code block. Keep it simple and readable.
Each code block should be able to print out the steps and results to ensure that you are aware of what has been done and what is left to be done.
For Bash scripts and commands, use the #!BASH marker at the beginning of your code block. This allows for both simple commands and multi-line scripts with variables, loops, conditionals, and other Bash features.

If you think the plan is finished, you should write a scientific report of the executed steps and results in a markdown format enclosed in "<report>...</report>" tags. If you have generated any visualizations, you should put them at appropriate places in the report with "![description](path/to/visualization)". If you have generated any output files, you should put the paths of the output files in the report by: "path/to/output_file": description.

In your response, you must include EITHER a <execute>...</execute> tag or a <report>...</report> tag. Do NOT include both or respond without either of them. Do NOT respond with empty messages.
"""

        if self.critic:
            self.prompt += """
You may receive feedback from the user or other agents. If so, you should carefully consider the feedback and update the plan accordingly.
"""
        self.prompt += "===============================\n"
        
        if self.tool_retriever:
            self.prompt += """
The environment supports a wide array of tools. You can retrieve tools from the tool retriever. The tool retriever receives a specific task description and returns a list of tools that are relevant to the task.
"""
            ### TODO: Add tool retriever
        else:
            self.prompt += """
Available tools:
"""            
            for tool in self.tools.available_tools():
                self.prompt += f"""
{tool}
---
{self.tools[tool].print_usage()}
---
"""
        if self.tool_call == "custom":
            self.prompt += """
You can call the tools using openbiomed functions. Some of the tools require molecule, protein, pocket, or text objects as input. Under these cases, you should create the object using the create_tool_input("data_type", "value") function. For example:
<execute>
from open_biomed.data import Molecule, Protein, Pocket, Text
from open_biomed.utils.misc import create_tool_input
molecule = create_tool_input("molecule", "C1=CC=C(C=C1)C(=O)O")   # supports .sdf, .pkl, and SMILES strings
protein = create_tool_input("protein", "./tmp/4xli.pdb")          # supports .pdb, .pkl, and FASTA strings
pocket = create_tool_input("pocket", "./tmp/4xli_pocket.pkl")     # supports .pkl files
text = create_tool_input("text", "Hello, world!")                 # supports text strings
</execute>

If you want to analyze the molecule, protein, pocket, or text objects, you can simply print the object to see the detailed information. You can also use the .smiles and .sequence attributes to get the SMILES and sequence strings of the molecule and protein objects.

Then, you can call the tools using the tool.run(input) function. For example:
<execute>
from open_biomed.tools.tool_registry import TOOLS
from open_biomed.utils.misc import create_tool_input
tool = TOOLS["molecule_property_prediction"]
molecule = create_tool_input("molecule", "C1=CC=C(C=C1)C(=O)O")
result, messages = tool.run(molecule=molecule, task="BBBP")
print(result[0])
</execute>
The tool will return two lists of the same length. The first list is the outputs, and the second list is the observations. The outputs are the results of the tool execution. The observations are a brief summary of the tool execution results. In the above example, 'result' is a python list of float numbers indicating the likelihood of the molecule to penetrate the blood-brain barrier. 'messages' is a list of strings representing the tool execution results.
"""
        else:
            self.prompt += """
You can call the tools using HTTP requests. For example:
<execute>
import requests
response = requests.post(f"{OPENBIOMED_SERVICE_URL}/tool/molecule_property_prediction", json={"molecule": "C1=CC=C(C=C1)C(=O)O", "task": "BBBP"})
</execute>
The tool will return a JSON object in the following format:
{
    "outputs": "output",
    "observations": "observation"
}
The outputs are the results of the tool execution (sometimes a remote url of the output file, which can be passed to the next tool as input). The observations are a brief summary of the tool execution results.

Some of the tools require molecule, protein, pocket, or text objects as input. Under these cases, you should pass a string either representing the file path or the SMILES/FASTA string of the molecule/protein.
- For molecules, you can pass the file path of the .sdf, .pkl, or SMILES string.
- For proteins, you can pass the file path of the .pdb, .pkl, or FASTA string.
- For pockets, you can pass the file path of the .pkl file.
- For text, you can pass the text string.
"""

        if getattr(self.memory, "workflow", False):
            if getattr(self.memory, "workflow_retriever", False):
                self.prompt += """
You can retrieve the predifined workflows that executes a specific composed task with multiple tool calls.
"""        
            else:
                self.prompt += """
You can use the predifined workflows that executes a specific composed task with multiple tool calls for multiple times. You should prepare the inputs for the workflow in a tuple list (step_id, input_name, input_value):
<execute>
from open_biomed.core.workflow import WORKFLOWS
workflow = WORKFLOWS["pdb_query"]
result, messages = workflow.run(inputs=[(1, "accession", "4xli"), (2, "accession", "4xli")], num_repeats=1)
print(result[0][0][0], result[0][1][0])
</execute>
The workflow will return a list of results and a list of messages. The `result[i][j][k]` are the results of the workflow execution for the k-th output of the j-th OUTPUT node of the i-th repeat. The `messages[i]` are the messages of the workflow execution for the i-th repeat.
"""
                for workflow_name, workflow in self.workflows.items():
                    self.prompt += f"""
Here are the descriptions, expected inputs, and expected outputs of the workflows:
Workflow {workflow_name}:
{yaml.dump(workflow.metadata)}
"""

        if "gpt" in self.llm_model_name or "openai" in self.llm_model_name:
            self.prompt += "\n\nIMPORTANT FOR GPT MODELS: You MUST use XML tags <execute> or <solution> in EVERY response. Do not use markdown code blocks (```) - use <execute> tags instead."

    def _setup_execution_environment(self):
        if self.use_docker:
            self.docker_client = aiodocker.Docker()
            work_dir = os.path.abspath(__file__).replace("open_biomed/core/agent.py", "")
            self.docker_container = self.docker_client.containers.run(
                config={
                    "Image": self.docker_container_id,
                    "Cmd": ["sleep", "infinity"],
                    "HostConfig": {"Binds": [f"{work_dir}:/workspace"]},
                    "WorkingDir": "/workspace",
                    "Tty": True,
                }
            )

    def _init_agent_workflow(self):
        def generate(state: AgentState) -> AgentState:
            message = [SystemMessage(content=self.prompt)] + state["messages"]
            response = self.llm.invoke(message)
            msg = str(response.content)
            if "<execute>" in msg and "</execute>" not in msg:
                msg += "</execute>"
            if "<report>" in msg and "</report>" not in msg:
                msg += "</report>"
            if "<think>" in msg and "</think>" not in msg:
                msg += "</think>"

            think_match = re.search(r"<think>(.*?)</think>", msg, re.DOTALL | re.IGNORECASE)
            execute_match = re.search(r"<execute>(.*?)</execute>", msg, re.DOTALL | re.IGNORECASE)
            report_match = re.search(r"<report>(.*?)</report>", msg, re.DOTALL | re.IGNORECASE)

            # Don't remove <think>...</think> blocks to avoid re-thinking
            # msg = re.sub(r"<think>.*?</think>", "", msg, flags=re.DOTALL | re.IGNORECASE)
            state["messages"].append(AIMessage(content=msg.strip()))
            state["num_steps"] += 1
            if report_match:
                state["next_step"] = "end"
            elif execute_match:
                state["next_step"] = "execute"
            elif think_match:
                state["next_step"] = "generate"
            else:
                state["chat_failure_count"] += 1
                if state["chat_failure_count"] >= self.chat_tolerance:
                    state["next_step"] = "end"
                    # Add a final message explaining the termination
                    state["messages"].append(
                        HumanMessage(
                            content="Execution terminated due to repeated parsing errors. Please check your input and try again."
                        )
                    )
                else:
                    state["next_step"] = "generate"
                    state["messages"].append(
                        HumanMessage(
                            content="The response must include a <think> ... </think> tag, a <execute> ... </execute> tag, or a <report> ... </report> tag, but there is none in your response. Please follow the instructions and generate a valid response again."
                        )
                    )
            return state

        def execute(state: AgentState) -> AgentState:
            msg = state["messages"][-1].content
            exec_match = re.search(r"<execute>(.*?)</execute>", msg, re.DOTALL | re.IGNORECASE)
            if exec_match:
                code = exec_match.group(1).strip()
                if code.startswith("#!BASH") or code.startswith("#!CLI"):
                    # bash scripts
                    code = code.replace("#!BASH", "").replace("#!CLI", "").strip()
                    if self.use_docker:
                        result = asyncio.run(exec_cmd_docker(code, self.docker_container, self.timeout))
                    else:
                        result = asyncio.run(exec_cmd(code, self.timeout))
                else:
                    # python code
                    code = code.strip("```").strip()
                    if self.use_docker:
                        result = asyncio.run(exec_python_docker(code, self.docker_container, self.timeout))
                    else:
                        global _persistent_namespace
                        result = asyncio.run(exec_python(code, self.timeout, _persistent_namespace))
                        dump_namespace(_persistent_namespace, f"{work_dir}/tmp/planner_executor-{self.thread_id}/namespace.pkl")
                        with open(f"{work_dir}/tmp/planner_executor-{self.thread_id}/captured_results.txt", "w") as f:
                            for captured in self.captured_results:
                                f.write(f"{captured}\n")

                if len(result[1]) > 10000:
                    result[1] = "The stdout is too long to display. Truncated by the first 10K characters.\n" + result[1][:10000]
                if len(result[2]) > 10000:
                    result[2] = "The stderr is too long to display. Truncated by the first 10K characters.\n" + result[2][:10000]

                if result[0] != 0:
                    state["code_failure_count"] += 1
                    if state["code_failure_count"] >= self.code_tolerance:
                        state["next_step"] = "generate"
                        state["code_failure_count"] = 0
                        msg = f"<observation>\nCode execution failed after {self.code_tolerance} debugging attempts.\n</observation>\n Please adjust your plan and try another action."
                    else:
                        state["next_step"] = "generate"
                        msg = f"<observation>\nCode execution failed with exit code {result[0]}.\n The stdout is:\n{result[1]}\nThe stderr is:\n{result[2]}\n</observation>\n Please check the stdout and stderr, and adjust your code."
                else:
                    state["next_step"] = "generate"
                    for i in range(state["code_failure_count"]):
                        state["messages"].pop(-2)
                        state["messages"].pop(-2)
                    state["code_failure_count"] = 0
                    msg = f"<observation>\nCode execution succeeded.\n The stdout is:\n{result[1]}\n</observation>\n Analyze the outputs and continue your task. If you think the current step is finished, remember to UPDATE THE CHECKLIST and show your thinking before executing the next step. NEVER generate the same code or attempt to recreate the outputs in the next step, as the tool execution results may be different."
            state["messages"].append(HumanMessage(content=msg))
            state["num_steps"] += 1
            return state

        app = StateGraph(AgentState)
        app.add_node("generate", generate)
        app.add_node("execute", execute)
        app.add_edge(START, "generate")
        app.add_conditional_edges(
            "generate", 
            lambda state: state["next_step"],
            {
                "generate": "generate",
                "execute": "execute",
                "end": END,
            }
        )
        app.add_edge("execute", "generate")
        if self.checkpointing:
            conn = sqlite3.connect(self.checkpointing_db_path, check_same_thread=False)
            self.checkpointer = SqliteSaver(conn)
            self.app = app.compile(checkpointer=self.checkpointer)
        else:
            self.app = app.compile()

    def run(self, user_prompt: str, thread_id: str = None):
        if thread_id is None:
            thread_id = get_thread_id("planner_executor")
            print("Starting new thread with ID:", thread_id)
            inputs = {"messages": [HumanMessage(content=user_prompt)], "num_steps": 0, "chat_failure_count": 0, "code_failure_count": 0, "next_step": None, "exec_namespace": {}}
            config = {"recursion_limit": 500, "configurable": {"thread_id": thread_id}}
            self.captured_results = []
        else:
            # import pdb; pdb.set_trace()
            print("Resuming thread with ID:", thread_id)
            inputs = None
            config = {"recursion_limit": 500, "configurable": {"thread_id": thread_id}}
            snapshot = self.app.get_state(config)
            for message in snapshot.values["messages"][:-1]:
                pretty_print(message)
            global _persistent_namespace
            _persistent_namespace = load_namespace(f"{work_dir}/tmp/planner_executor-{thread_id}/namespace.pkl")
            with open(f"{work_dir}/tmp/planner_executor-{thread_id}/captured_results.txt", "r") as f:
                self.captured_results = [line.strip() for line in f.readlines()]
        self.thread_id = thread_id
        logs = []
        global _patched
        _patched = False
        _apply_saving_patches("planner_executor", thread_id, self.captured_results)
        num_steps = 0
        for s in self.app.stream(inputs, stream_mode="values", config=config):
            # import pdb; pdb.set_trace()
            print(len(list(self.app.get_state_history(config))))
            message = s["messages"][-1].content
            logs.append(message)
            self.current_state = s
            pretty_print(s["messages"][-1])
            num_steps += 1
            # if num_steps > 10:
            #    exit(233)

        return logs, self.captured_results

    def export_report(self, format="markdown") -> str:
        msg = self.current_state["messages"][-1].content
        report_match = re.search(r"<report>(.*?)</report>", msg, re.DOTALL | re.IGNORECASE)
        if report_match:
            report_match = report_match.group(1).strip()
            for file in os.listdir(f"{work_dir}/tmp/planner_executor-{self.thread_id}"):
                if file not in report_match:
                    os.remove(f"{work_dir}/tmp/planner_executor-{self.thread_id}/{file}")
            report_match.replace(f"{work_dir}/tmp/planner_executor-{self.thread_id}/", "./")
            with open(f"{work_dir}/tmp/planner_executor-{self.thread_id}/report.md", "w") as f:
                f.write(report_match)
            if format == "pdf":
                try:
                    subprocess.run(["pandoc", f"{work_dir}/tmp/planner_executor-{self.thread_id}/report.md", "-o", f"{work_dir}/tmp/planner_executor-{self.thread_id}/report.pdf"], check=True)
                    return f"{work_dir}/tmp/planner_executor-{self.thread_id}/report.pdf"
                except Exception as e:
                    return f"Error: {e}"
            elif format == "markdown":
                return f"{work_dir}/tmp/planner_executor-{self.thread_id}/report.md"
        else:
            return None

    def export_as_workflow(self):
        pass

class DeepResearcher(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_tools(self):
        pass

    def _init_prompt(self):
        pass

    def run(self, user_prompt: str):
        pass

class Critic(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_tools(self):
        pass

    def _init_prompt(self):
        pass

    def run(self, user_prompt: str):
        pass

SUPPORTED_AGENTS = {
    "planner_executor": PlannerExecutor,
    "deep_researcher": DeepResearcher,
}

if __name__ == "__main__":
    cfg = Config(config_file=f"{work_dir}/configs/agent/default.yaml")
    agent = SUPPORTED_AGENTS[cfg.agent](cfg)
    logs, captured_results = agent.run(
        # "Design 10 novel drug candidates for the 4xli receptor. Each drug candidate should exhibit a Tanimoto similarity of at most 0.7 with other candidates.", 
        # thread_id="e01ed0e2-a8fa-46f2-ae96-206a5153367f"
        "Visualize the complex structure of chain A `./tmp/4xli.pdb`.",
    )
    report = agent.export_report()
    print(report)
    print(captured_results)