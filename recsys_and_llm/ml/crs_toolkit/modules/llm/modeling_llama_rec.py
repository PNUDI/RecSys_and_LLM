import logging
import os
import re

from crs_toolkit import BaseModule
from crs_toolkit.modules.monitor import monitor
from crs_toolkit.utility import DeviceManager
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

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, token=HF_TOKEN
        )
        # 모델과 토크나이저를 미리 로드 (매번 로드하지 않도록)
        self.tokenizer = self.get_tokenizer(model_name=model_name, token=HF_TOKEN)

        # <movie> 토큰 추가
        special_tokens_dict = {'additional_special_tokens': ['<movie>']}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))


        

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
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", **kwargs)
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
        print("[LlamaRec] raw_input:", raw_input)
        """
        주어진 raw_input에 대해 meta-llama/Llama-3.3-70B-Instruct 모델을 이용해
        인스트럭션에 맞는 응답을 생성합니다.
        """
        # topk가 0이면 빈 결과를 반환
        if topk == 0:
            return {"output": [], "links": []} if return_dict else []

        # 기본 max_tokens 값 설정 (없을 경우 512 토큰)
        max_tokens = max_tokens or 512

        # 외부에서 별도 토크나이저를 전달받지 않으면 미리 로드한 토크나이저 사용
        if tokenizer is None:
            tokenizer = self.tokenizer

        # 모델 선택 (별도로 지정하지 않으면 초기화 시 설정한 self.model_name 사용)
        model_used = model_name or self.model_name

        prompt_dict = self.prompt.copy()  # {'role': 'system','content':'...'}
        system_str = prompt_dict.get("content","")
        
        final_input = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system_str}\n"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"{raw_input}<|end_of_text|>"
        )

        # print("[LlamaRec] final input:", final_input)
        # 모델을 이용한 텍스트 생성
        tokenizer.pad_token_id = tokenizer.eos_token_id
        encodings = tokenizer(final_input, return_tensors="pt", padding=True)
        encodings = DeviceManager.copy_to_device(encodings, self.model.device)
        
        # inputs = tokenizer(final_prompt, return_tensors="pt", padding=True)
        # print("[LlamaRec] final input to model:", final_prompt)
        outputs = self.model.generate(
            **encodings,
            max_new_tokens=max_tokens,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("[LlamaRec] generated_text:", generated_text)

        cleaned_text = generated_text.strip()
        return cleaned_text
        # if return_dict:
        #     result = {"input": raw_input, "output": [cleaned_text]}
        #     print("[LlamaRec] returning dict:", result)
        #     return result
        # else:
        #     print("[LlamaRec] returning list:", [cleaned_text])
        #     return [cleaned_text]
