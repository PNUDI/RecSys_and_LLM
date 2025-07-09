# app/components/pipeline_manager.py
import os

from dotenv import load_dotenv

from recsys_and_llm.ml.crs_toolkit import (
    ChatgptAgent,
    ChatgptAgentConfig,
    ChatgptGen,
    ChatgptRec,
)

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
custom_prompt = {
    "role": "system",
    "content": """
    You are a system that recommends movies to users.
    You must answer in English,
    In the answer, the output format of Entity is
    **Limited to the movie title (opening year). **
    **However, exceptions are made when users ask for recommendation reasons or information about recommended movies.**
    If you lack information about your preferences (actors, directors, genres), etc., you can continue the conversation as much as you need.
    If the user shows signs of starting or ending the conversation, end the conversation with an appropriate greeting.
    """,
}

##################
# Recommendation #
##################
gpt_rec_fillblank = ChatgptRec.from_pretrained("recwizard/chatgpt-rec-fillblank")
##################
#   Generation   #
##################
gpt_rec = ChatgptRec.from_pretrained(
    "RecWizard/chatgpt-rec-fillblank", prompt=custom_prompt
)
gpt_gen = ChatgptGen.from_pretrained(
    "RecWizard/chatgpt-gen-fillblank", prompt=custom_prompt
)
gpt_gen_expansion = ChatgptGen.from_pretrained(
    model_name="gpt-4o",
    pretrained_model_name_or_path="recwizard/chatgpt-gen-expansion",
    prompt=custom_prompt,
)
gpt_gen_fillblank = ChatgptGen.from_pretrained(
    model_name="gpt-4o",
    pretrained_model_name_or_path="recwizard/chatgpt-gen-fillblank",
    prompt=custom_prompt,
)
##################
# Configurations #
##################
chat_gpt_agent_config = ChatgptAgentConfig()


def set_pipeline(flag, rec_module=gpt_rec, gen_module=gpt_gen):
    """
    flag, rec_module, gen_module에 따라 적절한 Pipeline 객체 생성.
    """
    return ChatgptAgent(
        rec_module=gpt_rec, gen_module=gpt_gen, config=chat_gpt_agent_config
    )
