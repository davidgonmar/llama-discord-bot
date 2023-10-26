from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
from llama_cpp import Llama
import replicate
from llama_discord_bot.util import run_async


@dataclass
class Message:
    """A message sent by a user or the bot."""
    content: str
    user: Literal["user", "bot"]


class LlamaBase(ABC):
    """Abstract base class for Llama models."""

    def __init__(self, system_prompt: str = ""):
        self.system_prompt = system_prompt

    @abstractmethod
    def generate_response(self, messages: list[Message], suffix: str = "") -> str:
        """Generate a response using the model."""

    def _generate_user_prompt(self, messages: list[Message]) -> str:
        """Generate a formatted user prompt."""
        user_messages = [
            message for message in messages if message.user in {"user", "bot"}]
        # Recommended way to format user input is by using the "[INST]" and "[/INST]" blocks,
        # while the bot's messages are not inside any block
        user_prompts = ["[INST]" + message.content + "[/INST]" if message.user ==
                        "user" else message.content for message in user_messages]

        return "\n".join(user_prompts)

    def _generate_prompt(self, messages: list[Message], suffix: str) -> str:
        """Generate a combined prompt for the model."""
        # Append the system prompt to the user prompt
        user_prompt = self._generate_user_prompt(messages=messages)
        return f"<<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{user_prompt}\n\n{suffix}"


class LlamaLocal(LlamaBase):
    """Uses llama_cpp locally to generate responses."""

    def __init__(self, model_path: str, system_prompt: str = ""):
        super().__init__(system_prompt)
        self.llama_cpp = Llama(model_path=model_path, n_ctx=2048)

    @run_async
    def generate_response(self, messages: list[Message], suffix: str = "") -> str:
        """Generate a response using the local model."""
        prompt = self._generate_prompt(messages=messages, suffix=suffix)
        llama_pred = self.llama_cpp.create_completion(prompt=prompt)
        text = llama_pred['choices'][0]['text']
        return text


class LlamaReplicate(LlamaBase):
    """Uses replicate (remote) to generate responses."""

    def __init__(self, replicate_model: str, system_prompt: str = ""):
        super().__init__(system_prompt)
        self.replicate_model = replicate_model

    @run_async
    def generate_response(self, messages: list[Message], suffix: str = "") -> str:
        """Generate a response using the replicate model."""
        input_data = {"prompt": self._generate_prompt(
            messages=messages, suffix=suffix)}
        output = replicate.run(self.replicate_model, input=input_data)
        return "".join(output)
