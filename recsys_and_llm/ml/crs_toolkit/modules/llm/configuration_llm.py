from crs_toolkit.configuration_utils import BaseConfig


class LLMConfig(BaseConfig):
    """
    The configuration of the generator based on OpenAI's GPT models.

    Attributes:
        answer_name (str): The special string used to represent the answer in the response.
        answer_mask (str): The type of the answer.
        prompt(str): The prompt for the GPT model.
        model_name(str): The specified GPT model's name.
    """

    def __init__(
        self,
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        max_gen_len: int = 0,
        answer_type="movie",
        answer_mask="<movie>",
        prompt: dict = None,
        **kwargs,
    ):
        """
        Initializes the instance of this configuration.

        Args:
            max_gen_len (int, optional): The maximum length to set in the generator.
        """

        super().__init__(**kwargs)
        self.model_name = model_name
        self.max_gen_len = max_gen_len
        self.answer_type = answer_type
        self.answer_mask = answer_mask
        if prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = {
                "role": "system",
                "content": (
                    " You are a customer support recommending movies to the user."
                    " Our operating platform returns suggested movies in real time from the dialogue history."
                    " You may choose from the suggestions and elaborate on why the user may like them."
                    " Or you can choose to reply without a recommendation."
                    " Now The system is suggesting the following movies: {}."
                    # ' Carefully review the dialogue history before I write a response to the user.'
                    ' If a movie comes in the format with a year, e.g. ""Inception (2010)"",'
                    " you should see the year (2010) as a part of the movie name."
                    ' You should not use the format ""Inception" (2010)" by leaving the year out of the quotation mark.'
                    " You should keep in mind that the system suggestion is only for reference."
                    # ' If the user is saying things like thank you or goodbye,'
                    " You should prioritize giving a quick short response over throwing more movies at the user,"
                    " especially when the user is likely to be leaving."
                ),
            }
