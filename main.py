import logging

import uvicorn
from fastapi import FastAPI

from app.router import router, set_dependencies
from app.safegpt_client import SafeGPTClient
from app.settings import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("proxy")

settings = Settings.load()

app = FastAPI(title="JetBrains OpenAI-Compatible Proxy", version="1.0.0")
app.include_router(router)

safegpt_client = SafeGPTClient(settings.safegpt_base_url, settings.safegpt_token)
set_dependencies(
    safegpt_client,
    settings.default_model_id,
    settings.web_search_chat_app_id,
    settings.code_interpreter_chat_app_id,
)

@app.get("/healthz")
def healthz():
    return {"ok": True, "service": "jetbrains-openai-compatible-proxy"}

def mask_secret(value: str) -> str:
    if not value:
        return "(not set)"
    return f"***{value[-4:]}" if len(value) > 12 else "***"

@app.on_event("startup")
def startup_event():
    shown = settings.model_dump()
    shown["safegpt_token"] = mask_secret(settings.safegpt_token)
    logger.info("Starting proxy with settings: %s", shown)

@app.on_event("shutdown")
async def shutdown_event():
    await safegpt_client.close()

if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
