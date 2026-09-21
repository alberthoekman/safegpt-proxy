import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

SESSION_TTL_SECONDS = 60 * 60 * 12

@dataclass
class SessionState:
    session_key: str
    conversation_id: Optional[str] = None
    model: str = "gpt-5.6-sol"
    chat_app_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)

    def touch(self):
        self.last_used_at = time.time()

SESSION_STORE: Dict[str, SessionState] = {}

def get_or_create_session(session_key: str, model: str, chat_app_ids: List[str]) -> SessionState:
    state = SESSION_STORE.get(session_key)
    if state is None:
        state = SessionState(
            session_key=session_key,
            model=model,
            chat_app_ids=chat_app_ids,
        )
        SESSION_STORE[session_key] = state
    else:
        state.model = model or state.model
        state.chat_app_ids = chat_app_ids or state.chat_app_ids
        state.touch()
    return state

def cleanup_sessions():
    cutoff = time.time() - SESSION_TTL_SECONDS
    stale = [k for k, v in SESSION_STORE.items() if v.last_used_at < cutoff]
    for k in stale:
        del SESSION_STORE[k]
