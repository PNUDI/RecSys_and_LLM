import logging
import os
from typing import List, Union


import openai
from crs_toolkit import BaseModule
from crs_toolkit.modules.monitor import monitor
from crs_toolkit.utility import DeviceManager
from transformers import AutoModelForCausalLM, AutoTokenizer

from .configuration_llm import LLMConfig
from .tokenizer_llama import LlamaTokenizer

logger = logging.getLogger(__name__)


class LlamaGen(BaseModule):
    """
    The generator implemented based on OpanAI's GPT models.

    """

    config_class = LLMConfig
    tokenizer_class = LlamaTokenizer

    def __init__(
        self, config: LLMConfig, prompt=None, model_name=None, debug=False, **kwargs
    ):
        """
        Initializes the instance based on the config file.

        Args:
            config (ChatgptGenConfig): The config file.
            prompt (str, optional): A prompt to override the prompt from config file.
            model_name (str, optional): The specified GPT model's name.
        """

        super().__init__(config, **kwargs)
        model_name = model_name or config.model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.prompt = config.prompt if prompt is None else prompt
        self.debug = debug
        self.tokenizer = self.get_tokenizer(model_name)

        # 커스텀 토큰 추가
        special_tokens_dict = {'additional_special_tokens': ['<movie>']}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, config=None, prompt=None, model_name=None
    ):
        """
        Get an instance of this class.

        Args:
            config:
            pretrained_model_name_or_path:
            prompt (str, optional): The prompt to override the prompt from config file.
            model_name (str, optional): The specified GPT model's name.

        Returns:
             the instance.
        """
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
        return cls(config, prompt=prompt, model_name=model_name)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        push_to_hub: bool = False,
        **kwargs,
    ):
        self.config.save_pretrained(
            save_directory=save_directory, push_to_hub=push_to_hub
        )

    @classmethod
    def get_tokenizer(cls, model_name=None, **kwargs):
        """
        Returns a tokenizer for the LlamaGen model.

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
        tokenizer,
        max_tokens=None,
        temperature=0.5,
        model_name=None,
        return_dict=False,
        **kwargs,
    ):
        """
        Generate a template to response the processed user's input.

        Args:
            raw_input (str): The user's raw input.
            tokenizer (BaseTokenizer, optional): A tokenizer to process the raw input.
            recs (list, optional): The recommended movies.
            max_tokens (int): The maximum number of tokens used for ChatGPT API.
            temperature (float): The temperature value used for ChatGPT API.
            model_name (str, optional): The specified GPT model's name.
            return_dict (bool): Whether to return a dict or a list.

        Returns:
            str: The template to response the processed user's input.
        """

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        prompt_dict = self.prompt.copy()  # {'role': 'system', 'content': '...'}
        system_str = prompt_dict.get('content', '')
        
        # Llama 형식의 프롬프트 생성 (Llama 3 양식 기준)
        # <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        # system 텍스트
        # <|start_header_id|>user<|end_header_id|>
        # 사용자 입력 <|eot_id|>

        final_input = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system_str}\n"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"{raw_input}<|eot_id|>"
        )

        tokenizer.pad_token = tokenizer.eos_token
        encodings = tokenizer(final_input, return_tensors="pt", padding=True)
        encodings = DeviceManager.copy_to_device(encodings, self.model.device)
        # encoding decode 테스트 해보기
        # print(tokenizer.decode(encodings['input_ids'][0], skip_special_tokens=True))

        max_tokens = max_tokens or 512

        res = self.model.generate(
            **encodings,
            max_new_tokens=max_tokens,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated_tokens = res[0][encodings['input_ids'].shape[-1]:]
        # print(f"Response: {res}")
        # print()
        # # decode only the first sequence to avoid list interpretation error
        # print(tokenizer.decode(generated_tokens, skip_special_tokens=True))
        # print()
        
        decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # "System:" 구문 제거, <movie>가 포함된 문장만 최대 3개 추출
        lines = decoded_text.split('\n')
        lines = [line for line in lines if "<movie>" in line]
        final_text = '\n'.join(lines)
        return final_text
