from crs_toolkit.configuration_utils import BaseConfig


class LlamaAgentConfig(BaseConfig):
    """
    The configuration for the Llama CRS model.

    Attributes:
        model_name (str): The name of the Llama model.
        prompt (str): The system prompt or instructions.
        answer_type (str): The type of answer tokens to generate.
    """

    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.prompt = ""
        self.answer_type = "movie"
