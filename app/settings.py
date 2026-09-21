from pydantic import BaseModel
import os

class Settings(BaseModel):
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    safegpt_base_url: str = os.getenv("SAFEGPT_BASE_URL", "https://api.safegpt.nl")
    safegpt_token: str = os.getenv("SAFEGPT_TOKEN",
                                   "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYW1laWQiOiJhN2QzNjA1OS1mZmRlLTQ3ZTAtYTk5ZS1mY2MwYTJkYTI0ZjAiLCJ1bmlxdWVfbmFtZSI6ImEuaG9la21hbkB2aWpmaGVlcmVubGFuZGVuLm5sIiwiZW1haWwiOiJhLmhvZWttYW5AdmlqZmhlZXJlbmxhbmRlbi5ubCIsInRva2VuX3R5cGUiOiJhcGlfYWNjZXNzIiwicm9sZSI6IlRlbmFudCBBZG1pbmlzdHJhdG9yIiwicGVybWlzc2lvbiI6WyIiLCJQRVJNX01PRFVMRV8qIiwiUEVSTV9URU5BTlRfKiIsIlBFUk1fVVNFUl8qIiwiUEVSTV9VU0VSX0NIQVRfQUNDRVNTIiwiUEVSTV9VU0VSX0lOU0lHSFRfQUNDRVNTIiwiUEVSTV9VU0VSX1NQRUNJQUxfKiIsIlBFUk1fVVNFUl9TUEVFQ0hfQUNDRVNTIiwiUEVSTV9VU0VSX1RFWFRfQUNDRVNTIiwiUEVSTV9VU0VSX1RSQU5TTEFURV9BQ0NFU1MiXSwibmJmIjoxNzg5NTU5NzIyLCJleHAiOjE4MzAyMTEyMDAsImlhdCI6MTc4OTU1OTcyMiwiaXNzIjoiaHR0cHM6Ly9hcHAuc2FmZWdwdC5ubCIsImF1ZCI6Imh0dHBzOi8vYXBpLnNhZmVncHQubmwifQ._DSaiDdYhlTxMniy-TLI3OijoPsKTKJowbRKSWxg5d0")

    default_model_id: str = os.getenv("DEFAULT_MODEL_ID", "gpt-5.6-sol")
    web_search_chat_app_id: str = os.getenv("SAFEGPT_WEB_SEARCH_CHAT_APP_ID", "48c7ce55-6208-4036-b6bb-69a0a1ab2b31")
    code_interpreter_chat_app_id: str = os.getenv("SAFEGPT_CODE_INTERPRETER_CHAT_APP_ID", "25144c6e-8e6d-4fb4-97c6-32ef2385c08d")
