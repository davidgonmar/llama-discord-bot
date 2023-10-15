from abc import ABC, abstractmethod
from llama_cpp import Llama as Llama
import replicate

class LlamaBase(ABC):
    @abstractmethod
    def generate_response(self, user_prompt: str):
        """Generate a response using the model."""
      

class LlamaLocal(LlamaBase):
    """Uses llama_cpp locally to generate responses."""
    def __init__(self, model_path: str):
        self.llama = Llama(model_path=model_path)

    def generate_response(self, user_prompt: str):
        """Generate a response using the local model."""
        llama_pred = self.llama(user_prompt)
        text = llama_pred.get('choices')[0].get('text')
        return text
    


class LlamaReplicate(LlamaBase):
    """Uses replicate (remote) to generate responses."""
    def __init__(self, replicate_model: str, system_prompt: str = ""):
        self.replicate_model = replicate_model
        self.system_prompt = system_prompt

    def generate_response(self, user_prompt):
        """Generate a response using the replicate model."""

        output = replicate.run(
            self.replicate_model,
            input={"prompt": user_prompt, "system_prompt": self.system_prompt})
            # Response comes chunked
        response_parts = []
        for item in output:
            response_parts.append(item)
        return "".join(response_parts)

       
        