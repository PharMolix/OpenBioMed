import asyncio
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
import logging
import os
from typing import List
from openai.types import Model

from open_biomed.models.foundation_models.biomedgpt import BioMedGPT4Chat, BioMedGPTR14Chat

load_dotenv(".env")
API_INFOS = {
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
    model: str="claude-sonnet-4-20250514",
    temperature: float=0.7,
    stop_sequences: List[str]=["\n\n"],
):
    if model[:7] == "claude-":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("langchain-anthropic is not installed. Please install it with `pip install langchain-anthropic`")
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            stop_sequences=stop_sequences,
            base_url=API_INFOS["ANTHROPIC_API_URL"],
            api_key=API_INFOS["ANTHROPIC_API_KEY"],
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
            base_url=API_INFOS["OPENAI_API_URL"],
            api_key=API_INFOS["OPENAI_API_KEY"],
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
            base_url=API_INFOS["GEMINI_API_URL"],
            api_key=API_INFOS["GEMINI_API_KEY"],
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
            base_url=API_INFOS["DEEPSEEK_API_URL"],
            api_key=API_INFOS["DEEPSEEK_API_KEY"],
        )
    elif model[:9] == "BioMedGPT":
        return CustomLLM(
            client=model,
            model_name_or_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", model),
            device="cpu",
        )

SUPPORTED_CUSTOM_LLMS = {
    "BioMedGPTR1": BioMedGPTR14Chat,
    "BioMedGPT": BioMedGPT4Chat,
}

class CustomLLM:
    def __init__(self,
        client: str,
        model_name_or_path: str,
        device: str="cpu",
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

    # TODO: Add LLM calls with ollama?