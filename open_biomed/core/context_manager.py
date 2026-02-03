from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import re
from typing import List, Optional

from open_biomed.utils.config import Config

class ContextManager:
    def __init__(self, config: Config):
        self.config = config
        self.context = []

    def reinitialize(self, system_prompt: Optional[SystemMessage]=None) -> None:
        self.context = []
        if system_prompt is not None:
            self.context.append(system_prompt)

    def add_message(self, message: BaseMessage, **kwargs) -> None:
        self.context.append(message)

    def get_last_message(self) -> BaseMessage:
        return self.context[-1]

    def get_context(self) -> List[BaseMessage]:
        return self.context

class ToolContextManager(ContextManager):
    def __init__(self, config: Config):
        super().__init__(config)
        self.raw_context = []
        self.to_drop_context = []

    def reinitialize(self, system_prompt: Optional[SystemMessage]=None) -> None:
        self.context = []
        self.raw_context = []
        self.to_drop_context = []
        if system_prompt is not None:
            self.context.append(system_prompt)
            self.raw_context.append(system_prompt)
            self.to_drop_context.append(False)

    def maintain(self) -> None:
        while self.to_drop_context[-1]:
            assert len(self.context) == len(self.to_drop_context)
            self.context.pop(-1)
            self.to_drop_context.pop(-1)

    def get_last_message(self) -> BaseMessage:
        return self.raw_context[-1]
        
    def add_message(self, message: BaseMessage, to_drop: bool=False, **kwargs) -> None:
        self.raw_context.append(message)
        if isinstance(message, AIMessage):
            if not to_drop:
                self.maintain()
            if isinstance(self.context[-1], AIMessage):
                self.context[-1].content += f"\n\n{message.content}"
            else:
                self.context.append(message)
                self.to_drop_context.append(to_drop)
        elif isinstance(message, HumanMessage):
            if self.config.merge_observation:
                obs_content = re.search(r"<observation>(.*?)</observation>", message.content, re.DOTALL | re.IGNORECASE)
                if obs_content is not None:
                    obs_content = obs_content.group(1)
                else:
                    obs_content = ""
                usr_content = message.content.replace(f"<observation>{obs_content}</observation>", "").strip()
                obs_content = obs_content.strip()
                if len(obs_content) > 0:
                    if isinstance(self.context[-1], AIMessage):
                        self.context[-1].content += f"\n\n<observation>{obs_content}</observation>"
                    else:
                        self.context.append(AIMessage(content=f"<observation>{obs_content}</observation>"))
                        self.to_drop_context.append(False)
                    self.context.append(HumanMessage(content=usr_content))
                    self.to_drop_context.append(True)
                else:
                    self.context.append(HumanMessage(content=usr_content))
                    self.to_drop_context.append(to_drop)
            else:
                self.context.append(message)
                self.to_drop_context.append(to_drop)
        else:
            self.context.append(message)
            self.to_drop_context.append(to_drop)

    def get_context(self) -> List[BaseMessage]:
        return self.context

CONTEXT_MANAGERS = {
    "default": ContextManager,
    "tool": ToolContextManager,
}