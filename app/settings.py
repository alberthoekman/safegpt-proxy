from pathlib import Path
import os
import tomllib

from pydantic import BaseModel, Field


class Settings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    safegpt_base_url: str = "https://api.safegpt.nl"
    safegpt_token: str = ""
    default_model_id: str = "gpt-5.6-sol"
    web_search_chat_app_id: str = "48c7ce55-6208-4036-b6bb-69a0a1ab2b31"
    code_interpreter_chat_app_id: str = "25144c6e-8e6d-4fb4-97c6-32ef2385c08d"

    @classmethod
    def load(cls) -> "Settings":
        config_path = Path("config.toml")
        data = {}

        if config_path.exists():
            with config_path.open("rb") as f:
                data = tomllib.load(f)

        return cls(
            host=os.getenv("HOST", data.get("host", "0.0.0.0")),
            port=int(os.getenv("PORT", data.get("port", 8000))),
            safegpt_base_url=os.getenv(
                "SAFEGPT_BASE_URL",
                data.get("safegpt_base_url", "https://api.safegpt.nl"),
            ),
            safegpt_token=os.getenv(
                "SAFEGPT_TOKEN",
                data.get("safegpt_token", ""),
            ),
            default_model_id=os.getenv(
                "DEFAULT_MODEL_ID",
                data.get("default_model_id", "gpt-5.6-sol"),
            ),
            web_search_chat_app_id=os.getenv(
                "SAFEGPT_WEB_SEARCH_CHAT_APP_ID",
                data.get("web_search_chat_app_id", "48c7ce55-6208-4036-b6bb-69a0a1ab2b31"),
            ),
            code_interpreter_chat_app_id=os.getenv(
                "SAFEGPT_CODE_INTERPRETER_CHAT_APP_ID",
                data.get("code_interpreter_chat_app_id", "25144c6e-8e6d-4fb4-97c6-32ef2385c08d"),
            ),
        )