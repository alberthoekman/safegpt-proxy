from pydantic import BaseModel
import os

class Settings(BaseModel):
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    safegpt_base_url: str = os.getenv("SAFEGPT_BASE_URL", "https://api.safegpt.nl")
    default_model_id: str = os.getenv("DEFAULT_MODEL_ID", "gpt-5.6-terra")
    web_search_chat_app_id: str = os.getenv("SAFEGPT_WEB_SEARCH_CHAT_APP_ID", "48c7ce55-6208-4036-b6bb-69a0a1ab2b31")
    code_interpreter_chat_app_id: str = os.getenv("SAFEGPT_CODE_INTERPRETER_CHAT_APP_ID", "25144c6e-8e6d-4fb4-97c6-32ef2385c08d")
