from abc import ABC, abstractmethod
from llama_cpp import Llama
import replicate
from util import run_async

class LlamaBase(ABC):
    """Abstract base class for llama models."""

    system_prompt: str

    @abstractmethod
    def generate_response(self, user_prompt: str):
        """Generate a response using the model."""

    def _generate_prompt(self, user_prompt: str):
        """Given a user prompt and a system prompt, generate a prompt for the model."""

        return f"<<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{user_prompt}"
      

class LlamaLocal(LlamaBase):
    """Uses llama_cpp locally to generate responses."""
    
    def __init__(self, model_path: str, system_prompt: str = ""):
        self.llama_cpp = Llama(model_path=model_path, n_ctx=2048)
        self.system_prompt = system_prompt

    @run_async
    def generate_response(self, user_prompt: str):
        """Generate a response using the local model."""

        llama_pred = self.llama_cpp.create_completion(
            prompt=self._generate_prompt(user_prompt=user_prompt),
        )
        text = llama_pred.get('choices')[0].get('text')

        return text
    


class LlamaReplicate(LlamaBase):
    """Uses replicate (remote) to generate responses."""
    def __init__(self, replicate_model: str, system_prompt: str = ""):
        self.replicate_model = replicate_model
        self.system_prompt = system_prompt

    @run_async
    def generate_response(self, user_prompt):
        """Generate a response using the replicate model."""

        output = replicate.run(
            self.replicate_model,
            input={"prompt": self._generate_prompt(user_prompt=user_prompt)},
        )

        # Response comes chunked
        return "".join(output)

       
        