# app/components/pipeline_manager.py

def set_pipeline(flag, rec_module, gen_module):
    """
    flag, rec_module, gen_module에 따라 적절한 Pipeline 객체 생성.
    """
    if flag.lower() == "blank":
        return "FillBlankPipeline"
    elif flag.lower() == "expansion":
        return "ExpansionPipeline"
    elif flag.lower() == "gpt":
        return "GptPipeline"
    return "DefaultUniCRS"
