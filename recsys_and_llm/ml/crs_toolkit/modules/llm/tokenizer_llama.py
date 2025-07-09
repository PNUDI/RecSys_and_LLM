from typing import Optional

from crs_toolkit.tokenizer_utils import BaseTokenizer
from crs_toolkit.utility import SEP_TOKEN
from transformers import AutoTokenizer
from typing_extensions import override


class LlamaTokenizer(BaseTokenizer):
    """
    The tokenizer for the generator based on OpenAI's GPT models.
    """

    def __init__(self, **kwargs):
        """
        Initializes the instance of this tokenizer.
        """
        super().__init__(
            tokenizers=[
                AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct"),
            ]
        )
        self.tokenizers[0].pad_token = self.tokenizers[0].eos_token
        self.tokenizers[0].pad_token_id = self.tokenizers[0].eos_token_id

    @property
    def eos_token(self):
        return self.tokenizers[0].eos_token

    @property
    def eos_token_id(self) -> int:
        return self.tokenizers[0].eos_token_id

    @override
    def preprocess(self, raw_input, **kwargs):
        """
        Process the raw input by extracting the pure text.

        Args:
            context (str): The raw input.

        Returns:

        """

        texts = raw_input.split(SEP_TOKEN)
        user = "User:"
        system = "System:"
        context = ""
        for text in texts:
            if text.startswith(system):
                context += text[len(system) :].strip(" ") + "</s>"
            else:
                context += "<s>" + "[INST]" + text[len(user) :].strip(" ") + "[/INST]"

        return context
