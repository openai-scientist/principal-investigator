"""File store of consultations: the redacted request, the status and the result."""

import json
import os
import re
from pathlib import Path
from typing import Any

from principal_investigator.contract import ConsultationResult

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, consultation_id: str) -> Path:
        if not _SAFE_ID.fullmatch(consultation_id) or consultation_id in {".", ".."}:
            raise ValueError("invalid consultation id")
        return self.root / f"{consultation_id}.json"

    def read(self, consultation_id: str) -> dict[str, Any] | None:
        path = self._path(consultation_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def write(self, consultation_id: str, record: dict[str, Any]) -> None:
        path = self._path(consultation_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)

    def create(self, consultation_id: str, request: dict[str, Any]) -> None:
        self.write(consultation_id, {"status": "running", "request": request, "result": None})

    def finish(self, consultation_id: str, result: ConsultationResult) -> None:
        record = self.read(consultation_id) or {"request": None}
        record["status"] = result.status
        record["result"] = result.model_dump(mode="json")
        self.write(consultation_id, record)

    def all_ids(self) -> list[str]:
        return [p.stem for p in self.root.glob("*.json")]
