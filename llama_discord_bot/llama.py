from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
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

    def _generate_prompt(self, messages: list[Message], suffix: str) -> str:
        """From a list of messages, generate a prompt, including both the user and system prompts."""

        # The prompting technique is the one from Meta's paper.
        # A good explanation can be found at https://huggingface.co/blog/llama2#how-to-prompt-llama-2

        if (len(messages) == 0):
            raise ValueError("Cannot generate prompt from empty messages list")

        # We need to group messages into groups like:
        # [[user, user, user....], [bot, bot, bot...], [user, user, user...], ...]
        # basically group consecutive messages from the same author together
        # we dont know what the first message is, so we need to find out

        current_role = messages[0].user
        messages_grouped = [[messages[0]]]

        for message in messages[1:]:
            if message.user == current_role:
                messages_grouped[-1].append(message)
            else:
                messages_grouped.append([message])
                current_role = message.user

        # append all messages for each sub-list into a single string,
        #  so for each message group user= messages[0].user and content = all messages in the group
        messages = [Message(group[0].user, "".join(
            [message.content for message in group])) for group in messages_grouped]

        # now, for each pair, format it like
        # <s>
        # {{ bot_message }}
        # [INST] <<SYS>>
        # {{ system_prompt }}
        # <</SYS>>
        # {{ user_message }} [/INST] </s>

        # if there is no bot or user message, simply write "" System prompt and their delimiters only appended on first pair

        prompt = ""
        i = 0
        while i < len(messages):
            message = messages[i]
            bot_message = message.content.strip(
            ) if message.user == "bot" else ""
            user_message = message.content.strip(
            ) if message.user == "user" else ""

            if (user_message):
                if (len(messages) > i + 1 and messages[i + 1].user == "bot"):
                    # Dont do i + 1 since its done later
                    bot_message = messages[i + 1].content.strip()

            system_prompt = (self.SYSTEM_INPUT_START +
                             self.system_prompt + self.SYSTEM_INPUT_END) if (i == 0 and self.system_prompt.strip() != "") else ""
            prompt += f"""<s>
            [INST]
            {system_prompt}
            {user_message}
            [/INST]
            {bot_message}
            </s>"""
            if bot_message != "":
                i += 1
            if user_message != "":
                i += 1

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
