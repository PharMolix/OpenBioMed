import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import ast
import asyncio
import logging
import shutil
from datetime import datetime
from typing import List, Optional, TypedDict, AsyncGenerator, Tuple
from typing_extensions import Self
from openai import OpenAI, Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta

from open_biomed.data import Text
from open_biomed.utils.config import Config
from open_biomed.tools.base_tool import Tool
from open_biomed.tools.web_request_tools import WebSearchRequester
from open_biomed.core.email_server import EmailServer
from open_biomed.models.foundation_models.biomedgpt import BioMedGPT4Chat, BioMedGPTR14Chat

from dotenv import load_dotenv

load_dotenv(".env")
API_INFOS = {
        "api_key": os.getenv("API_KEY"),
        "api_url": os.getenv("API_URL"),
        "model_name": os.getenv("MODEL_NAME")
    }

class StreamChunk(TypedDict):
    final_resp: str
    reasoning: str

class ContextDict(TypedDict):
    ref_text: str
    others: dict


class LLM():
    def __init__(self) -> None:
        pass
    
    def _init_client(self) -> Self:
        # Init LLM client from local path or API
        raise NotImplementedError
    
    def _update_query(self, query: str, context: ContextDict = {"ref_text": "", "others": dict()}) -> str:
        # Update query with additional contexts, including references or others.
        raise NotImplementedError
    
    def _get_input(self, query: str, context: ContextDict = {"ref_text": "", "others": dict()}):
        # Format messages with query and context
        raise NotImplementedError
    
    def generate(self, query: str, context: ContextDict = {"ref_text": "", "others": dict()}):
        # Generate answers with client
        raise NotImplementedError
    

class LLM_Local(LLM):
    def __init__(self,
        client_name: str,
        model_name_or_path: str,
        device: Optional[str]=None
    ) -> None:

        super(LLM_Local, self).__init__()

        self.supported_clients = {
            "BioMedGPTR1": BioMedGPTR14Chat,
            "BioMedGPT": BioMedGPT4Chat
        }

        try:
            self.client_model = self.supported_clients[client_name]
        except:
            raise ValueError("Only support BioMedGPTR1 and BioMedGPT for now.")

        self._init_client(api_infos)
    
    def _init_client(self, model_name_or_path: str, device: Optional[str]=None) -> Self:
        self.client = self.client_model.from_pretrained(model_name_or_path=model_name_or_path, device=device)
    
    def _update_query(self, query: str, context: ContextDict = {"ref_text": "", "others": dict()}) -> str:
        return f"Answer the question based on the following: {context['ref_text']}\n" + " " + query
    
    def _get_input(self, query: str, context: ContextDict = {"ref_text": "", "others": dict()}, is_debug=False):
        query = self._update_query(query=query, context=context) if context['ref_text'] else query
        return query
    
    def generate(self, query: str, context: ContextDict = {"ref_text": "", "others": dict()}, is_debug=False):
        text_input = self._get_input(query=query, context=context, is_debug=is_debug)
        resp = self.client.chat(Text.from_str(text_input))
        return resp
    

class LLM_API(LLM):
    def __init__(self,
        api_infos: dict,
        temperature: float = 0.0
    ) -> None:

        super(LLM_API, self).__init__()

        self._init_client(api_infos)
        
        self.model_name = api_infos['model_name']
        self.temperature = temperature

        self.think_start, self.think_end = "<think>", "</think>"
    
    def _init_client(self, api_infos: dict) -> Self:

        self.client = OpenAI(
            api_key=api_infos['api_key'],
            base_url=api_infos['api_url']
            )

    
    def _update_query(self, query: str, context: ContextDict = {"ref_text": "", "others": dict()}) -> str:
        return f"Answer the question based on the following: {context['ref_text']}\n" + " " + query
    
    def _get_input(self, query: str, context: ContextDict = {"ref_text": "", "others": dict()}, is_debug=False):
        messages = []
        query = self._update_query(query=query, context=context) if context['ref_text'] else query
        messages.append({"role": "user", "content": query})

        if is_debug:
            logging.info("[LLM Input] " + str(messages))

        return messages

    async def generate_stream(self, query: str, context: ContextDict = {"ref_text": "", "others": dict()}, is_debug=False):

        messages = self._get_input(query=query, context=context, is_debug=is_debug)

        stream: Stream[ChatCompletionChunk] = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
        )

        is_think = False
        for chunk in stream:
            delta: ChoiceDelta = chunk.choices[0].delta
            delta_json = delta.model_dump()
            content = delta_json.get("content", "")
            if content != "" or content!=None:
                if content == self.think_start:
                    is_think = True
                    continue
                elif content == self.think_end:
                    is_think = False
                    continue
                elif is_think:
                    stream_chunk: StreamChunk = {"final_resp": "", "reasoning": content}
                else:
                    stream_chunk: StreamChunk = {"final_resp": content, "reasoning": ""}
            yield stream_chunk

    def generate(self, query: str, context: ContextDict = {"ref_text": "", "others": dict()}, is_debug=False):

        messages = self._get_input(query=query, context=context, is_debug=is_debug)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False,
            temperature=self.temperature
        )

        text=response.choices[0].message.content
        start_index = text.find(self.think_start) + len(self.think_start)
        end_index = text.find(self.think_end)
        resp_thinking = text[start_index:end_index].strip()

        resp_final = text[:start_index - len(self.think_start)] + text[end_index + len(self.think_end):].strip()
        return {
            "final_resp": resp_final,
            "reasoning": resp_thinking
        }

class LLMExtractor(LLM_API):
    def __init__(self, api_infos: dict, temperature: float = 0.0):
        super(LLMExtractor, self).__init__(api_infos=api_infos, temperature=temperature)

    def _update_query(self, query: str, context: ContextDict = {"ref_text": "", "others": dict()}):
        query = f"Here are some references, complete the task based on them\n---\n{context['ref_text']}\n---\nTask: {context['others']['task_desp']}"
        return query

class LLMReportGenerator(LLM_API):
    def __init__(self, api_infos: dict, temperature: float = 0.0):
        super(LLMReportGenerator, self).__init__(api_infos=api_infos, temperature=temperature)

    def _update_query(self, query: str, context: ContextDict = {"ref_text": "", "others": dict()}):
        query = f"Here are some references, complete the task based on them\n---\n{context['ref_text']}\n---\n" + \
            f"Task: Generate a scientific report for biomedical expert according to the references. The report name is '{context['others']['title']}'. The report should be detailed and follows the following structure:\n---\n{context['others']['structure']}"
        return query

class KeyInfoExtractor(Tool):

    def __init__(self, api_infos: dict=API_INFOS) -> None:
        from open_biomed.tools.web_request_tools import WebSearchRequester
        self.agent_search = WebSearchRequester()
        self.agent_extractor = LLMExtractor(api_infos=api_infos, temperature=0.0)
        self.task_prompts = {
            "sbdd": {
                "search": "{}",
                "extract": "Extract all the mentioned molecule names (or PubChem IDs), protein names (or target names), and optimization intentions. Your output should be in the following format: {'molecules': [''], 'proteins': [''], 'intentions': ['']}.",
            },
            "default": {
                "search": "{}疾病的已有药物",
                "extract": "Extract all the mentioned drugs' molecule (name, SMILES, or PubChem IDs). Your output should be in the following format: {'drugs': ['']}."
            }
        }

    def print_usage(self) -> str:
        return "\n".join([
            'Search relevent information of the query and extract key values from that for workflow.',
            'Inputs: {"query": query to be searched}',
            'Outputs: A dict of list. Each list contains the extracted values.'
        ])
    
    def run(self, query: str, task: str="default") -> Tuple[list[str], str]:
        
        try:
            task_desp = self.task_prompts[task]["extract"]
        except:
            raise ValueError(f"{task} is currently not supported!")
        
        logging.info("[Retrival] Start")

        retrivals = self.agent_search.run(query=self.task_prompts[task]["search"].format(query))[0][0]

        logging.info("[Retrival] Done")

        context = {
            "ref_text": retrivals,
            "others": {
                "task_desp": task_desp
            }
        }

        question = ""

        resp = self.agent_extractor.generate(query=question, context=context, is_debug=True)

        try:
            result = eval(resp['final_resp'])
        except:
            result = ['ibuprofen']
        
        reasoning = resp['reasoning']

        return result, reasoning

class ReportGeneratorGeneral(Tool):
    def __init__(self, api_infos: dict=API_INFOS) -> None:
        from open_biomed.core.oss_warpper import Oss_Warpper

        self.current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.dir_path = f"./tmp/temp_{self.current_time}"
        os.mkdir(self.dir_path)

        self.title = "OpenBioMed平台工作流报告"
        self.structure = "\n## 1. 工作流设计（对实验涉及的输入输出和使用的工具进行简要描述）\n" + \
                            "## 2. 工作流结果与分析（对实验产生的结果进行描述，并进行适当的推理分析）\n" + \
                            "## 3. 总结\n" + \
                                "### 3.1 基础信息（逐点列出实验日期、生成模型为ChatDD）\n" + \
                                "### 3.2 分析总结（对报告全文进行总结）\n"
        self.ref_infos = f"\n\n实验时间：{self.current_time}\n" + \
                        "生成模型名称：ChatDD（不要写明版本号）\n" + \
                        "报告生成单位：北京水木分子生物科技有限公司\n"

        self.email_subject = f"工作流报告_{self.current_time}"
        self.email_body = "尊敬的用户，您好，\n感谢您使用OpenBioMed平台，您在平台提交的工作流已自动完成，其涉及到的输出结果文件和实验报告请参考邮件附件。"

        self.llm = LLMReportGenerator(api_infos=api_infos, temperature=0.0)
        self.oss_warpper = Oss_Warpper()
        self.email_server = EmailServer()
        self.email_try_max = 3
        
    
    def print_usage(self) -> str:
        return "\n".join([
            'Report generation for general task.',
            'Inputs: {"pipeline": pipeline for general task}',
            'Outputs: a report for general task.'
        ])
    
    async def _run_workflow(self, workflow: str, num_repeats: int, user_email: str):
        from open_biomed.core.workflow import Workflow, parse_frontend

        config_file = parse_frontend(workflow)
        try:
            config = Config(config_file=config_file)
            workflow = Workflow(config)
            await workflow.run(num_repeats=num_repeats, context=open(f"{self.dir_path}/workflow_outputs.txt", "w"), tool_outputs=open(f"{self.dir_path}/workflow_tool_outputs.txt", "w"))
        except:
            # Please ensure that your workflow can run successfully before submitting
            for i in range(self.email_try_max):
                is_email_success = self.email_server.send(user_email=user_email, \
                                        subject=self.email_subject, \
                                        body=config_file, \
                                        timestamp=self.current_time)
                if not is_email_success:
                    break
                else:
                    logging.info(f"[Email] Retrying...")
                    self.email_server = EmailServer()

    
    def _gen_context(self, output_file):
        with open(output_file, "r") as f:
            references = f.read()
        context = {
            "ref_text": references + self.ref_infos,
            "others":{
                "title": self.title,
                "structure": self.structure
            }
        }
        return context
    
    def _save_zip_outputs(self, report):
        import subprocess

        with open(f"{self.dir_path}/workflow_tool_outputs.txt", "r") as f:
            tool_outputs = f.read().split("\n")[:-1]
        os.remove(f"{self.dir_path}/workflow_tool_outputs.txt")

        for file in tool_outputs:
            file_name = os.path.basename(file)
            dst_path = os.path.join(self.dir_path, file_name)
            os.rename(file, dst_path)
        with open(os.path.join(self.dir_path, "report.md"), 'w') as f:
            f.write(report['final_resp'])
        subprocess.run(f"zip -r {self.dir_path}/{self.email_subject}.zip {self.dir_path}", shell=True, check=True)
    
    def _oss_upload(self):
        oss_file_path = self.oss_warpper.generate_file_name(f"{self.dir_path}/{self.email_subject}.zip")
        self.oss_warpper.upload(oss_file_path, f"{self.dir_path}/{self.email_subject}.zip")

    
    async def run(self, workflow: str, user_email: str, num_repeats: int) -> str:
        # Workflow
        await self._run_workflow(workflow=workflow, num_repeats=num_repeats, user_email=user_email)

        # Report generation
        context = self._gen_context(output_file=f"{self.dir_path}/workflow_outputs.txt")       
        question = ""
        resp = self.llm.generate(query=question, context=context, is_debug=True)

        # Save & zip
        self._save_zip_outputs(report=resp)
        
        # OSS uploading
        self._oss_upload()

        # Email
        for i in range(self.email_try_max):
            is_email_success = self.email_server.send(user_email=user_email, subject=self.email_subject, body=self.email_body, attachment_path=f"{self.dir_path}/{self.email_subject}.zip", timestamp=self.current_time)

            if is_email_success:
                # Temporal files cleaning
                if os.path.exists(self.dir_path):
                    shutil.rmtree(self.dir_path)
                break
            else:
                logging.info(f"[Email] Retrying...")
                self.email_server = EmailServer()

        return resp

class ReportGeneratorSBDD(ReportGeneratorGeneral):
    def __init__(self, api_infos: dict=API_INFOS) -> None:
        super(ReportGeneratorSBDD, self).__init__(api_infos=api_infos)

        self.title = "基于靶点的分子设计报告"
        self.structure = "\n## 1. 靶点基本信息介绍\n" + \
                                "### 1.1 靶点简介（详细介绍靶点的生物学功能）\n" + \
                                "### 1.2 靶点与疾病的关联性\n" + \
                                    "1.2.1 靶点-疾病的相关信号通路\n" + \
                                    "1.2.2 靶点研究的临床意义与现状\n" + \
                                "### 1.3 靶点结构特征\n" + \
                                    "1.3.1 蛋白结构\n" + \
                                    "1.3.2 活性位点\n" + \
                                    "1.3.3 已知配体\n" + \
                                "### 1.4 靶点研究的临床意义与现状\n" + \
                            "## 2. 分子生成和虚拟筛选（首先分析初始生成分子，形成2.1子章节，然后逐个分子优化组合进行分析，形成2.2、2.3等子章节）\n" + \
                                "### 2.1 初始生成分子分析\n" + \
                                    "2.1.1 生成分子分析\n" + \
                                        "A. 生成分子SMILES\n" + \
                                        "B. 生成分子与靶点口袋的亲和力\n" + \
                                        "C. 生成分子结构特征（从关键官能团和药效团进行逐点分析描述）\n" + \
                                    "2.1.2 成药性预测\n" + \
                                        "A. 理化性质分析（以表格形式展示总logP、分子量和氢键供体受体等指标，并进行详细文字分析描述）\n" + \
                                        "B. 毒性分析（以表格形式展示关键指标，并进行详细分析描述，如hERG毒性和CYP抑制）\n" + \
                                "### 2.2 优化分子1分析（对比优化分子和初始分子，如果优化后亲和力下降，则不展示和分析）\n" + \
                                    "2.2.1 生成分子分析\n" + \
                                        "A. 优化后分子SMILES\n" + \
                                        "B. 优化分子与靶点口袋的亲和力（对比初始分子和优化分子亲和力，并进行文字分析）\n" + \
                                        "C. 优化分子结构特征（从关键官能团和构效关系进行逐点分析描述）\n" + \
                                    "2.2.2 成药性预测\n" + \
                                        "A. 理化性质分析（以表格形式展示总logP、分子量和氢键供体受体等指标，并进行详细文字分析描述）\n" + \
                                        "B. 毒性分析（以表格形式展示关键指标，并进行详细分析描述，如hERG毒性和CYP抑制）\n" + \
                                "### 2.3 优化分子2分析（如果有，则同上）\n" + \
                            "## 3. 总结\n" + \
                                "### 3.1 实验基础信息（逐点列出实验日期、生成模型为ChatDD、实验涉及的靶点名和分子SMILES、实验涉及的Tools）\n" + \
                                "### 3.2 实验分析总结（对报告全文进行总结）\n"
        self.ref_ranges =  "\n\n| 性质 | 标准范围 | 说明 |\n" + \
                            "|---|---|---|\n" + \
                            "| 分子量 (MW) | < 500 | 较小的分子量有利于药物的吸收和渗透。|\n" + \
                            "| LogP | < 5 | 表示脂水分配系数，反映分子的亲脂性，数值越小越有利于水溶性。|\n" + \
                            "| 氢键供体数 (HBD) | ≤ 5 | 氢键供体数量过多会影响药物的渗透性。|\n" + \
                            "| 氢键受体数 (HBA) | ≤ 10| 氢键受体数量过多可能导致药物在细胞膜中的滞留。|\n" + \
                            "| 可旋转键数 (RB)  | ≤ 10| 可旋转键数过多会影响药物的刚性，进而影响其生物活性。|\n" + \
                            "| 拓扑极性表面积 (TPSA) | < 140 Å² | 用于评估分子的极性表面，数值越大，分子的极性越强。|\n" 

        self.email_subject = f"基于靶点的分子设计报告_{self.current_time}"     
    
    def print_usage(self) -> str:
        return "\n".join([
            'Report generation for SBDD task.',
            'Inputs: {"pipeline": pipeline for SBDD task}',
            'Outputs: a report for SBDD task.'
        ])
    
    def _gen_context(self, output_file):
        with open(output_file, "r") as f:
            references = f.read()
        context = {
            "ref_text": references + self.ref_ranges + self.ref_infos,
            "others":{
                "title": self.title,
                "structure": self.structure
            }
        }
        return context



if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    agent = ReportGeneratorSBDD()
    asyncio.run(agent.run(workflow="", user_email="", num_repeats=5))




