import pytest
from llama_discord_bot.llama import LlamaBase, Message


# Allows to instantiate LlamaBase (abstract class)
LlamaBase.__abstractmethods__ = set()


class TestLlamaBase():

    def compare_strings_without_spaces(self, a, b, rel_tol=1e-9, abs_tol=0.0):
        a = a.replace(" ", "").replace("\n", "").replace("\t", "")
        b = b.replace(" ", "").replace("\n", "").replace("\t", "")
        return pytest.approx(a, rel=rel_tol, abs=abs_tol) == b

    def test_system_prompt_bot_first(self):
        # pylint: disable=abstract-class-instantiated
        llama = LlamaBase(
            "You are a helpful, respectful and honest assistant.")
        test_messages = [
            Message("bot", "Hello, how are you?"),
            Message("user", "I'm fine, thanks!"),
            Message("bot", "That's good to hear!"),
        ]

        prompt = llama._generate_prompt(test_messages, "Hello, how are you?")

        expected_prompt = """
        <s>
        [INST]
        <<SYS>> You are a helpful, respectful and honest assistant. <</SYS>>
        [/INST]
        Hello, how are you?
        </s>
        <s>
        [INST]
        I'm fine, thanks!
        [/INST]
        That's good to hear!
        </s>
        """

        assert self.compare_strings_without_spaces(prompt, expected_prompt)

    def test_no_system_prompt_bot_first(self):
        # pylint: disable=abstract-class-instantiated
        llama = LlamaBase()
        test_messages = [
            Message("bot", "Hello, how are you?"),
            Message("user", "I'm fine, thanks!"),
            Message("bot", "That's good to hear!"),
        ]

        prompt = llama._generate_prompt(test_messages, "")

        expected_prompt = """
        <s>
        [INST]
        [/INST]
        Hello, how are you?
        </s>
        <s>
        [INST]
        I'm fine, thanks!
        [/INST]
        That's good to hear!
        </s>
        """

        assert self.compare_strings_without_spaces(prompt, expected_prompt)

    def test_system_prompt_user_first(self):
        # pylint: disable=abstract-class-instantiated
        llama = LlamaBase(
            "You are a helpful, respectful and honest assistant.")
        test_messages = [
            Message("user", "Hello, how are you?"),
            Message("bot", "I'm fine, thanks!"),
            Message("user", "That's good to hear!"),
        ]

        prompt = llama._generate_prompt(test_messages, "Hello, how are you?")

        expected_prompt = """
        <s>
        [INST]
        <<SYS>> You are a helpful, respectful and honest assistant. <</SYS>>
        Hello, how are you?
        [/INST]
        I'm fine, thanks!
        </s>
        <s>
        [INST]
        That's good to hear!
        [/INST]
        </s>
        """

        assert self.compare_strings_without_spaces(prompt, expected_prompt)

    def test_no_system_prompt_user_first(self):
        # pylint: disable=abstract-class-instantiated
        llama = LlamaBase()
        test_messages = [
            Message("user", "Hello, how are you?"),
            Message("bot", "I'm fine, thanks!"),
            Message("user", "That's good to hear!"),
        ]

        prompt = llama._generate_prompt(test_messages, "")

        expected_prompt = """
        <s>
        [INST]
        Hello, how are you?
        [/INST]
        I'm fine, thanks!
        </s>
        <s>
        [INST]
        That's good to hear!
        [/INST]
        </s>
        """
        assert self.compare_strings_without_spaces(prompt, expected_prompt)
