import re
import sys

from crs_toolkit.model_utils import BasePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

from .configuration_llama_agent import LlamaAgentConfig
from crs_toolkit.modules.llm import LlamaRec
from crs_toolkit.modules.llm import LlamaGen


class LlamaAgent(BasePipeline):
    """
    The CRS model based on Meta's Llama models.

    Attributes:
        config (LlamaAgentConfig): The Llama agent config.
        model_name (str): The specified Llama model's name.
    """

    config_class = LlamaAgentConfig

    def __init__(self, config, rec_module, gen_module, model_name=None, temperature=1.0, **kwargs):
        super().__init__(config, gen_module=gen_module, rec_module=rec_module, **kwargs)

        self.model_name = config.model_name if model_name is None else model_name
        self.prompt_template = config.prompt or ""
        self.temperature = temperature

        # Hugging Face transformers를 이용해 토크나이저와 모델을 로드합니다.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        if rec_module is None:
            self.rec_module = LlamaRec.from_pretrained(model_name or config.model_name)
        else:
            self.rec_module = rec_module

        if gen_module is None:
            self.gen_module = LlamaGen.from_pretrained(model_name or config.model_name)
        else:
            self.gen_module = gen_module

        self.rec_tokenizer = self.rec_module.get_tokenizer()
        self.gen_tokenizer = self.gen_module.get_tokenizer()
        self.movie_pattern = re.compile(r"<movie>")
        self.answer_type = "movies"

    def response(self, query, max_tokens=128, **kwargs):
        # 1) 우선 Gen 모듈을 사용해 1차 응답 생성
        gen_output = self.gen_module.response(query, tokenizer=self.gen_tokenizer)
        # 2) 1차 응답에서 영화 개수 파악 후 Rec 모듈로 추천 생성
        n_movies = len(self.movie_pattern.findall(gen_output if isinstance(gen_output, str) else gen_output[0]))
        
        rec = self.rec_module.response(gen_output, topk=n_movies, tokenizer=self.rec_tokenizer)
        print("[LlamaAgent] rec output:", rec)
        pure_output = gen_output.replace("System:", "").strip()
        # prompt_template을 사용하지 않고, 입력된 출력 그대로 사용
        replaced_text = pure_output
        # 4) Hugging Face 모델로 최종 응답 생성
        inputs = self.tokenizer(replaced_text, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=512,  # 기존보다 크게 설정
            temperature=self.temperature,
            do_sample=True,
            top_k=5,
            **kwargs,
        )
        final_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return f"Final Answer:\n{final_text}"
