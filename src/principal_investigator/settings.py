"""Service settings from the environment. Model keys come with each request, never from here."""

import os
from dataclasses import dataclass
from pathlib import Path

VERSION = "0.1.0"


@dataclass(frozen=True)
class Settings:
    service_key: str
    store_dir: Path
    host: str = "127.0.0.1"
    port: int = 8003
    llm_timeout_seconds: float = 180.0
    #: Sent to OpenAlex as ``mailto`` so lookups use its polite pool.
    contact_email: str | None = None
    #: OpenAlex API key; with it, lookups draw on the key's own quota, not the shared pool.
    openalex_api_key: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        key = os.environ.get("PI_SERVICE_KEY", "")
        if len(key) < 32:
            raise RuntimeError("PI_SERVICE_KEY must be set to at least 32 characters")
        return cls(
            service_key=key,
            store_dir=Path(os.environ.get("PI_STORE_DIR", "runs")),
            host=os.environ.get("PI_HOST", "127.0.0.1"),
            port=int(os.environ.get("PI_PORT", "8003")),
            llm_timeout_seconds=float(os.environ.get("PI_LLM_TIMEOUT_SECONDS", "180")),
            contact_email=os.environ.get("PI_CONTACT_EMAIL") or None,
            openalex_api_key=os.environ.get("PI_OPENALEX_API_KEY") or None,
        )
