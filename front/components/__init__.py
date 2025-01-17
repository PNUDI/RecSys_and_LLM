# front/components/__init__.py
from .conversation_manager import save_conversation, load_conversation
from .pipeline_manager import set_pipeline
from .response_generator import get_unicrs_response
from .recommend_main import rec_main, rec_line