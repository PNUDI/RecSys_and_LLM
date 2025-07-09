import logging
import os

from crs_toolkit import BaseModule
from crs_toolkit.modules.monitor import monitor
from transformers import AutoModelForCausalLM, AutoTokenizer

from .configuration_llm_rec import LLMRecConfig
from .tokenizer_llama import LlamaTokenizer

logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_AUTH_TOKEN")


class LlamaRec(BaseModule):
    """
    The recommender based on Llama models.
    """

    config_class = LLMRecConfig
    tokenizer_class = LlamaTokenizer

    def __init__(
        self,
        config: LLMRecConfig,
        prompt=None,
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        debug=False,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.model_name = config.model_name if model_name is None else model_name
        self.prompt = config.prompt if prompt is None else prompt
        self.debug = debug

        # 모델과 토크나이저를 미리 로드 (매번 로드하지 않도록)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=HF_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, token=HF_TOKEN
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, config=None, prompt=None, model_name=None
    ):
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
        return cls(config, prompt=prompt, model_name=model_name)

    def save_pretrained(self, save_directory, push_to_hub=False, **kwargs):
        self.config.save_pretrained(
            save_directory=save_directory, push_to_hub=push_to_hub
        )

    @classmethod
    def get_tokenizer(cls, model_name=None, **kwargs):
        """
        Returns a tokenizer for the LlamaRec model.

        Args:
            model_name (str): The model name or path to load the tokenizer from.
                              If None, use the default from the class config.

        Returns:
            tokenizer (PreTrainedTokenizer): Loaded tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct", **kwargs
        )
        return tokenizer

    @monitor
    def response(
        self,
        raw_input,
        tokenizer=None,
        topk=3,
        max_tokens=None,
        temperature=0.7,
        model_name=None,
        return_dict=False,
        **kwargs,
    ):
        """
        주어진 raw_input에 대해 meta-llama/Llama-3.3-70B-Instruct 모델을 이용해
        인스트럭션에 맞는 응답을 생성합니다.
        """
        # topk가 0이면 빈 결과를 반환
        if topk == 0:
            return {"output": [], "links": []} if return_dict else []

        # 기본 max_tokens 값 설정 (없을 경우 128 토큰)
        max_tokens = max_tokens or 128

        # 외부에서 별도 토크나이저를 전달받지 않으면 미리 로드한 토크나이저 사용
        if tokenizer is None:
            tokenizer = self.tokenizer

        # 모델 선택 (별도로 지정하지 않으면 초기화 시 설정한 self.model_name 사용)
        model_used = model_name or self.model_name

        # meta-llama/Llama-3.3-70B-Instruct 모델 카드 권장 인스트럭션 프롬프트 형식 사용
        prompt_template = self.prompt or (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            "Instruction: {input}\n\nResponse:"
        )
        prompt_text = prompt_template.format(input=raw_input)

        # 모델을 이용한 텍스트 생성
        inputs = tokenizer(prompt_text, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_k=topk,
            **kwargs,
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if return_dict:
            return {"input": raw_input, "output": [generated_text], "links": []}
        else:
            return [generated_text]
