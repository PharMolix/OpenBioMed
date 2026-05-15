import asyncio
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
import logging
import os
import shutil
from datetime import datetime
from typing import List

from open_biomed.models.foundation_models.biomedgpt import BioMedGPT4Chat, BioMedGPTR14Chat

load_dotenv(".env")

# All LLM API configurations
# Priority: API_KEY + API_URL > Platform providers (by model prefix)
API_INFOS = {
    # Custom/Self-hosted LLM (OpenAI-compatible API)
    # If API_KEY and API_URL are configured, this takes priority
    "API_KEY": os.getenv("API_KEY"),
    "API_URL": os.getenv("API_URL"),
    "MODEL_NAME": os.getenv("MODEL_NAME"),

    # Platform providers - only used if API_KEY/API_URL are NOT configured
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    "ANTHROPIC_API_URL": os.getenv("ANTHROPIC_API_URL"),

    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "OPENAI_API_URL": os.getenv("OPENAI_API_URL"),

    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    "GEMINI_API_URL": os.getenv("GEMINI_API_URL"),

    "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY"),
    "DEEPSEEK_API_URL": os.getenv("DEEPSEEK_API_URL"),
}


def get_llm(
    model: str = None,
    temperature: float = None,
    stop_sequences: List[str] = ["\n\n"],
):
    # Track if model was explicitly provided by user/config
    explicit_model = model is not None

    if model is None:
        model = API_INFOS.get("MODEL_NAME") or "claude-sonnet-4-20250514"
    if temperature is None:
        temperature = 0.0

    # Priority 1: If model is NOT explicitly provided (uses env MODEL_NAME)
    # and API_KEY + API_URL are configured, use custom LLM
    # This allows MODEL_NAME to be anything while using your custom API
    if not explicit_model and API_INFOS.get("API_KEY") and API_INFOS.get("API_URL"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("langchain-openai is not installed. Please install it with `pip install langchain-openai`")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            stop_sequences=stop_sequences,
            base_url=API_INFOS.get("API_URL"),
            api_key=API_INFOS.get("API_KEY"),
        )

    # Priority 2: Match by prefix to platform providers (or explicit model)
    if model[:7] == "claude-":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("langchain-anthropic is not installed. Please install it with `pip install langchain-anthropic`")
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            stop_sequences=stop_sequences,
            base_url=API_INFOS.get("ANTHROPIC_API_URL"),
            api_key=API_INFOS.get("ANTHROPIC_API_KEY"),
        )
    elif model[:7] == "openai-":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("langchain-openai is not installed. Please install it with `pip install langchain-openai`")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            stop_sequences=stop_sequences,
            base_url=API_INFOS.get("OPENAI_API_URL"),
            api_key=API_INFOS.get("OPENAI_API_KEY"),
        )
    elif model[:7] == "gemini-":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("langchain-openai is not installed. Please install it with `pip install langchain-openai`")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            stop_sequences=stop_sequences,
            base_url=API_INFOS.get("GEMINI_API_URL"),
            api_key=API_INFOS.get("GEMINI_API_KEY"),
        )
    elif model[:9] == "deepseek-":
        try:
            from langchain_deepseek import ChatDeepSeek
        except ImportError:
            raise ImportError("langchain-deepseek is not installed. Please install it with `pip install langchain-deepseek`")
        return ChatDeepSeek(
            model=model,
            temperature=temperature,
            stop_sequences=stop_sequences,
            base_url=API_INFOS.get("DEEPSEEK_API_URL"),
            api_key=API_INFOS.get("DEEPSEEK_API_KEY"),
        )
    elif model[:9] == "BioMedGPT":
        return CustomLLM(
            client=model,
            model_name_or_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", model),
            device="cpu",
        )
    else:
        # Custom/Self-hosted LLM (OpenAI-compatible API)
        # Fallback if API_KEY/API_URL not configured but model doesn't match any prefix
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("langchain-openai is not installed. Please install it with `pip install langchain-openai`")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            stop_sequences=stop_sequences,
            base_url=API_INFOS.get("API_URL"),
            api_key=API_INFOS.get("API_KEY"),
        )


SUPPORTED_CUSTOM_LLMS = {
    "BioMedGPTR1": BioMedGPTR14Chat,
    "BioMedGPT": BioMedGPT4Chat,
}


class CustomLLM:
    def __init__(self,
        client: str,
        model_name_or_path: str,
        device: str = "cpu",
    ):
        try:
            self.client = SUPPORTED_CUSTOM_LLMS[client]
        except:
            raise ValueError(f"Unsupported LLM: {client}")
        self.client = SUPPORTED_CUSTOM_LLMS
        if not os.path.exists(model_name_or_path):
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            logging.info("Repo not found. Try downloading from snapshot")
            if client == "BioMedGPT":
                repo_id = "PharMolix/BioMedGPT-LM-7B"
            elif client == "BioMedGPTR1":
                repo_id = "PharMolix/BioMedGPTR1-R1"
            snapshot_download(repo_id=repo_id, local_dir=model_name_or_path, force_download=True)
        self.device = device

    def __call__(self, *args, **kwargs):
        return self.client(*args, **kwargs)


# ============== Report Generator Classes ==============
class ReportGeneratorGeneral:
    def __init__(self, model: str = None, temperature: float = None) -> None:
        from open_biomed.core.oss_warpper import Oss_Warpper
        from open_biomed.core.email_server import EmailServer

        self.current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.dir_path = f"./tmp/temp_{self.current_time}"
        os.makedirs(self.dir_path, exist_ok=True)

        self.title = "OpenBioMed 平台工作流报告"
        self.structure = "\n## 1. 工作流设计（对实验涉及的输入输出和使用的工具进行简要描述）\n" + \
                            "## 2. 工作流结果与分析（对实验产生的结果进行描述，并进行适当的推理分析）\n" + \
                            "## 3. 总结\n" + \
                                "### 3.1 基础信息（逐点列出实验日期、生成模型为 ChatDD）\n" + \
                                "### 3.2 分析总结（对报告全文进行总结）\n"
        self.ref_infos = f"\n\n实验时间：{self.current_time}\n" + \
                        "生成模型名称：ChatDD（不要写明版本号）\n" + \
                        "报告生成单位：北京水木分子生物科技有限公司\n"

        self.email_subject = f"工作流报告_{self.current_time}"
        self.email_body = "尊敬的用户，您好，\n感谢您使用 OpenBioMed 平台，您在平台提交的工作流已自动完成，其涉及到的输出结果文件和实验报告请参考邮件附件。"

        # Use get_llm() instead of custom LLM_API class
        self.llm = get_llm(model=model, temperature=temperature)
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
        from open_biomed.utils.config import Config

        config_file = parse_frontend(workflow)
        try:
            config = Config(config_file=config_file)
            workflow = Workflow(config)
            await workflow.run(num_repeats=num_repeats,
                              context=open(f"{self.dir_path}/workflow_outputs.txt", "w"),
                              tool_outputs=open(f"{self.dir_path}/workflow_tool_outputs.txt", "w"))
        except Exception as e:
            logging.error(f"[Workflow] Error: {e}")
            for i in range(self.email_try_max):
                is_email_success = self.email_server.send(user_email=user_email,
                                        subject=self.email_subject,
                                        body=config_file,
                                        timestamp=self.current_time)
                if not is_email_success:
                    break
                else:
                    logging.info(f"[Email] Retrying...")
                    self.email_server = EmailServer()

    def _gen_context(self, output_file):
        with open(output_file, "r") as f:
            references = f.read()
        return references

    def _build_report_prompt(self, references: str) -> str:
        return f"""Here are some references from the workflow execution:
---
{references}
---
Task: Generate a scientific report for biomedical expert according to the references. The report name is '{self.title}'. The report should be detailed and follows the following structure:
{self.structure}

Please write the report in Chinese."""

    def _parse_llm_response(self, response) -> dict:
        """Parse langchain LLM response to expected format"""
        try:
            # Langchain returns a string or AIMessage
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)

            # Try to extract thinking tags if present
            think_start = "<think>"
            think_end = "</think>"
            start_index = content.find(think_start) + len(think_start)
            end_index = content.find(think_end)

            if start_index > len(think_start) - 1 and end_index > start_index:
                resp_thinking = content[start_index:end_index].strip()
                resp_final = content[:start_index - len(think_start)] + content[end_index + len(think_end):].strip()
            else:
                resp_thinking = ""
                resp_final = content.strip()

            return {
                "final_resp": resp_final,
                "reasoning": resp_thinking
            }
        except Exception as e:
            logging.error(f"Error parsing LLM response: {e}")
            return {
                "final_resp": str(response),
                "reasoning": ""
            }

    def _save_zip_outputs(self, report):
        import subprocess

        with open(f"{self.dir_path}/workflow_tool_outputs.txt", "r") as f:
            tool_outputs = f.read().split("\n")[:-1]
        os.remove(f"{self.dir_path}/workflow_tool_outputs.txt")

        for file in tool_outputs:
            file_name = os.path.basename(file)
            dst_path = os.path.join(self.dir_path, file_name)
            if os.path.exists(file):
                os.rename(file, dst_path)
        with open(os.path.join(self.dir_path, "report.md"), 'w') as f:
            f.write(report['final_resp'])
        subprocess.run(f"zip -r {self.dir_path}/{self.email_subject}.zip {self.dir_path}", shell=True, check=True)

    def _oss_upload(self):
        oss_file_path = self.oss_warpper.generate_file_name(f"{self.dir_path}/{self.email_subject}.zip")
        self.oss_warpper.upload(oss_file_path, f"{self.dir_path}/{self.email_subject}.zip")

    async def run(self, workflow: str, user_email: str, num_repeats: int):
        await self._run_workflow(workflow=workflow, num_repeats=num_repeats, user_email=user_email)

        references = self._gen_context(output_file=f"{self.dir_path}/workflow_outputs.txt")
        prompt = self._build_report_prompt(references)

        # Use langchain's invoke method
        from langchain.schema import HumanMessage
        response = self.llm.invoke([HumanMessage(content=prompt)])
        resp = self._parse_llm_response(response)

        self._save_zip_outputs(report=resp)
        self._oss_upload()

        for i in range(self.email_try_max):
            is_email_success = self.email_server.send(user_email=user_email,
                                                      subject=self.email_subject,
                                                      body=self.email_body,
                                                      attachment_path=f"{self.dir_path}/{self.email_subject}.zip",
                                                      timestamp=self.current_time)
            if is_email_success:
                if os.path.exists(self.dir_path):
                    shutil.rmtree(self.dir_path)
                break
            else:
                logging.info(f"[Email] Retrying...")
                self.email_server = EmailServer()

        return resp


class ReportGeneratorSBDD(ReportGeneratorGeneral):
    def __init__(self, model: str = None, temperature: float = None) -> None:
        super(ReportGeneratorSBDD, self).__init__(model=model, temperature=temperature)

        self.title = "基于靶点的分子设计报告"
        self.structure = "\n## 1. 靶点基本信息介绍\n" + \
                                "### 1.1 靶点简介（详细介绍靶点的生物学功能）\n" + \
                                "### 1.2 靶点与疾病的关联性\n" + \
                                    "1.2.1 靶点 - 疾病的相关信号通路\n" + \
                                    "1.2.2 靶点研究的临床意义与现状\n" + \
                                "### 1.3 靶点结构特征\n" + \
                                    "1.3.1 蛋白结构\n" + \
                                    "1.3.2 活性位点\n" + \
                                    "1.3.3 已知配体\n" + \
                                "### 1.4 靶点研究的临床意义与现状\n" + \
                            "## 2. 分子生成和虚拟筛选（首先分析初始生成分子，形成 2.1 子章节，然后逐个分子优化组合进行分析，形成 2.2、2.3 等子章节）\n" + \
                                "### 2.1 初始生成分子分析\n" + \
                                    "2.1.1 生成分子分析\n" + \
                                        "A. 生成分子 SMILES\n" + \
                                        "B. 生成分子与靶点口袋的亲和力\n" + \
                                        "C. 生成分子结构特征（从关键官能团和药效团进行逐点分析描述）\n" + \
                                    "2.1.2 成药性预测\n" + \
                                        "A. 理化性质分析（以表格形式展示总 logP、分子量和氢键供体受体等指标，并进行详细文字分析描述）\n" + \
                                        "B. 毒性分析（以表格形式展示关键指标，并进行详细分析描述，如 hERG 毒性和 CYP 抑制）\n" + \
                                "### 2.2 优化分子 1 分析（对比优化分子和初始分子，如果优化后亲和力下降，则不展示和分析）\n" + \
                                    "2.2.1 生成分子分析\n" + \
                                        "A. 优化后分子 SMILES\n" + \
                                        "B. 优化分子与靶点口袋的亲和力（对比初始分子和优化分子亲和力，并进行文字分析）\n" + \
                                        "C. 优化分子结构特征（从关键官能团和构效关系进行逐点分析描述）\n" + \
                                    "2.2.2 成药性预测\n" + \
                                        "A. 理化性质分析（以表格形式展示总 logP、分子量和氢键供体受体等指标，并进行详细文字分析描述）\n" + \
                                        "B. 毒性分析（以表格形式展示关键指标，并进行详细分析描述，如 hERG 毒性和 CYP 抑制）\n" + \
                                "### 2.3 优化分子 2 分析（如果有，则同上）\n" + \
                            "## 3. 总结\n" + \
                                "### 3.1 实验基础信息（逐点列出实验日期、生成模型为 ChatDD、实验涉及的靶点名和分子 SMILES、实验涉及的 Tools）\n" + \
                                "### 3.2 实验分析总结（对报告全文进行总结）\n"
        self.ref_ranges = "\n\n| 性质 | 标准范围 | 说明 |\n" + \
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

    def _build_report_prompt(self, references: str) -> str:
        return f"""Here are some references from the workflow execution:
---
{references}
---
{self.ref_ranges}
---
Task: Generate a scientific report for biomedical expert according to the references. The report name is '{self.title}'. The report should be detailed and follows the following structure:
{self.structure}

Please write the report in Chinese."""
