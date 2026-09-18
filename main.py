import logging
from fastapi import FastAPI
import uvicorn

from app.settings import Settings
from app.router import router, set_dependencies
from app.safegpt_client import SafeGPTClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("proxy")

settings = Settings()

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

@app.on_event("startup")
def startup_event():
    logger.info("Starting proxy with settings: %s", settings.model_dump())

@app.on_event("shutdown")
async def shutdown_event():
    await safegpt_client.close()

if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
