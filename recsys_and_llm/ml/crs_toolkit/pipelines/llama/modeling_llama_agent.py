import re
import sys

from crs_toolkit.model_utils import BasePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

from .configuration_llama_agent import LlamaAgentConfig


class LlamaAgent(BasePipeline):
    """
    The CRS model based on Meta's Llama models.

    Attributes:
        config (LlamaAgentConfig): The Llama agent config.
        model_name (str): The specified Llama model's name.
    """

    config_class = LlamaAgentConfig

    def __init__(self, config, model_name=None, temperature=1.0, **kwargs):
        super().__init__(config, **kwargs)
        self.model_name = config.model_name if model_name is None else model_name
        self.temperature = temperature

        # Hugging Face transformers를 이용해 토크나이저와 모델을 로드합니다.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    def response(self, query, max_tokens=128, **kwargs):
        """
        Generate agent response.

        Args:
            query (str): The user's input.

        Returns:
            A response object with the generated text content.
        """
        # Meta-Llama 모델 카드 권장 인스트럭션 프롬프트 템플릿
        prompt_template = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            "Instruction: {input}\n\nResponse:"
        )
        prompt_text = prompt_template.format(input=query)

        # 프롬프트를 토크나이징
        inputs = self.tokenizer(prompt_text, return_tensors="pt")

        # 모델을 사용해 텍스트 생성
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            temperature=self.temperature,
            do_sample=True,
            top_k=5,  # 필요에 따라 top_k 값을 조정할 수 있습니다.
            **kwargs,
        )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 간단한 응답 객체 생성
        class AgentResponse:
            def __init__(self, content):
                self.content = content

            def __str__(self):
                return self.content

            def __repr__(self):
                return f"AgentResponse(content={self.content})"

        return AgentResponse(generated_text)
