from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
from llama_cpp import Llama
import replicate
from llama_discord_bot.run_async import run_async


@dataclass
class Message:
    """A message sent by a user or the bot."""

    content: str
    user: Literal["user", "bot"]


class LlamaBase(ABC):
    """Abstract base class for Llama models."""

    # Constants used to format the user input
    USER_INPUT_START = "[INST]"
    USER_INPUT_END = "[/INST]"
    SYSTEM_INPUT_START = "<<SYS>>"
    SYSTEM_INPUT_END = "<</SYS>>"

    def __init__(self, system_prompt: str = ""):
        self.system_prompt = system_prompt

    @abstractmethod
    def generate_response(self, messages: list[Message], suffix: str = "") -> str:
        """Generate a response using the model."""

    def _generate_user_prompt(self, messages: list[Message]) -> str:
        """Generate a formatted user prompt from a list of messages."""

        # Format user input using "[INST]" and "[/INST]" blocks, while bot's messages remain unchanged
        user_prompts = [self.USER_INPUT_START + message.content + self.USER_INPUT_END if message.user ==
                        "user" else message.content for message in messages]

        return "\n".join(user_prompts)

    def _generate_prompt(self, messages: list[Message], suffix: str) -> str:
        """From a list of messages, generate a prompt, including both the user and system prompts."""

        # Append the system prompt to the user prompt
        user_prompt = self._generate_user_prompt(messages=messages)
        return f"{self.SYSTEM_INPUT_START}\n{self.system_prompt}\n{self.SYSTEM_INPUT_END}\n\n{user_prompt}\n\n{suffix}"


class LlamaLocal(LlamaBase):
    """Uses llama_cpp locally to generate responses."""

    def __init__(self, model_path: str, system_prompt: str = ""):
        super().__init__(system_prompt)
        self.llama_cpp = Llama(model_path=model_path, n_ctx=2048)

    @run_async
    def generate_response(self, messages: list[Message], suffix: str = "") -> str:
        """Generates a response using a local model. Uses llama.cpp under the hood."""

        completion = self.llama_cpp.create_completion(
            prompt=self._generate_prompt(messages=messages, suffix=suffix))

        return completion['choices'][0]['text']


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

        # Replicate returns data separated into chunks, so we need to join them
        output = replicate.run(self.replicate_model, input=input_data)
        return "".join(output)
