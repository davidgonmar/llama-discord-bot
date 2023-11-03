from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
from itertools import groupby
from llama_cpp import Llama
import replicate
from llama_discord_bot.run_async import run_async


@dataclass
class Message:
    """A message sent by a user or the bot."""
    user: Literal["user", "bot"]
    content: str


class LlamaBase(ABC):
    """Abstract base class for Llama models."""

    # Constants used to format the user input
    USER_INPUT_START = "[INST]"
    USER_INPUT_END = "[/INST]"
    SYSTEM_INPUT_START = "<<SYS>>"
    SYSTEM_INPUT_END = "<</SYS>>"
    MESSAGES_PAIR_START = "<s>"
    MESSAGES_PAIR_END = "</s>"

    def __init__(self, system_prompt: str = ""):
        self.system_prompt = system_prompt

    @abstractmethod
    def generate_response(self, messages: list[Message], suffix: str = "") -> str:
        """Generate a response using the model."""

    def _merge_consecutive_messages_by_role(self, messages: list[Message]) -> list[Message]:
        """Merges consecutive messages from the same author into a single message with concatenated content"""

        def keyfunc(message): return message.user

        # Group messages by author
        messages_grouped = [list(group)
                            for key, group in groupby(messages, keyfunc)]

        # Merge all the content from each group into a single message
        messages = [Message(group[0].user, "".join(
            message.content for message in group)) for group in messages_grouped]

        return messages

    def _generate_prompt(self, messages: list[Message], suffix: str) -> str:
        """From a list of messages, generate a prompt, including both the user and system prompts. It uses the prompting technique from Meta's paper."""

        if not messages:
            raise ValueError("Cannot generate prompt from empty messages list")

        # Merge consecutive messages from the same author into a single message with concatenated content
        messages = self._merge_consecutive_messages_by_role(messages)

        # Initialize the prompt as an empty string
        prompt = ""
        i = 0
        while i < len(messages):
            message = messages[i]
            bot_message = message.content.strip() if message.user == "bot" else ""
            user_message = message.content.strip() if message.user == "user" else ""

            # If the next message is from the bot, use its content for the bot message
            if user_message and (i + 1 < len(messages)) and (messages[i + 1].user == "bot"):
                bot_message = messages[i + 1].content.strip()

            # Add the system prompt only for the first message and if it's not empty
            system_prompt = (self.SYSTEM_INPUT_START +
                             self.system_prompt + self.SYSTEM_INPUT_END) if (i == 0 and self.system_prompt.strip() != "") else ""

            # Add the prompt part to the string
            prompt += f"""<s>
            [INST]
            {system_prompt}
            {user_message}
            [/INST]
            {bot_message}
            </s>"""

            i += 1  # Increment the index to move to the next message
            if (bot_message and user_message):
                i += 1  # Increment the index to skip the bot message in the next iteration if we used 2 messages

        if (suffix):
            prompt += f"""
            <s>
            [INST]
            {suffix}
            [/INST]
            </s>
            """
        return prompt.strip()


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
