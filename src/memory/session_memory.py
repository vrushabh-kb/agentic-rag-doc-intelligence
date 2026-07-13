"""
In-session conversational memory. Per your call: session-only, no
cross-session persistence, so this is a plain in-memory store keyed by
session_id - no database needed. If you later want cross-session memory,
this is the one class you'd swap for a Redis/SQLite-backed version; nothing
else in the codebase needs to change.
"""
from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class SessionState:
    turns: list[Turn] = field(default_factory=list)
    last_used_doc_ids: list[str] | None = None


class SessionMemoryStore:
    def __init__(self):
        self._sessions: dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState()
        return self._sessions[session_id]

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        self.get(session_id).turns.append(Turn(role, content))

    def set_last_used_docs(self, session_id: str, doc_ids: list[str] | None) -> None:
        self.get(session_id).last_used_doc_ids = doc_ids

    def has_context(self, session_id: str) -> bool:
        return len(self.get(session_id).turns) > 0

    def history_as_text(self, session_id: str, max_turns: int = 6) -> str:
        """Last N turns formatted for prompt injection."""
        turns = self.get(session_id).turns[-max_turns:]
        if not turns:
            return ""
        return "\n".join(f"{t.role.upper()}: {t.content}" for t in turns)

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


# Single shared instance imported by both the FastAPI app and the orchestrator.
memory_store = SessionMemoryStore()
