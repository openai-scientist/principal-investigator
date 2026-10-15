"""The artifacts of one consultation, fetched from the Platform only when the agent reads them."""

import hashlib
import json
import re
from typing import Any

import httpx

from principal_investigator.contract import SERVICE_KEY_HEADER, ConsultArtifact

PAGE = 8_000
_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)


class ArtifactError(RuntimeError):
    pass


class Workspace:
    def __init__(
        self,
        refs: list[ConsultArtifact],
        service_key: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        current_stage_run: str | None = None,
    ):
        self.refs = {r.artifact_id: r for r in refs}
        #: The stage run the consultation is about; artifacts of other runs are history.
        self.current_stage_run = current_stage_run
        self.service_key = service_key
        self.transport = transport
        self._cache: dict[str, str] = {}
        self.read_ids: set[str] = set()

    def catalog(self) -> str:
        if not self.refs:
            return "(no artifacts)"
        lines = []
        for r in self.refs.values():
            size = f", {r.size_bytes} bytes" if r.size_bytes is not None else ""
            attempt = ""
            if self.current_stage_run and r.stage_run_id:
                attempt = (
                    ", current attempt"
                    if r.stage_run_id == self.current_stage_run
                    else ", earlier attempt"
                )
            lines.append(
                f"- {r.artifact_id}: {r.schema_id} v{r.version}, role={r.role}, "
                f"stage={r.stage_key or '-'}{attempt}{size}"
            )
        return "\n".join(lines)

    async def text(self, artifact_id: str) -> str:
        if artifact_id in self._cache:
            return self._cache[artifact_id]
        ref = self.refs.get(artifact_id)
        if ref is None:
            raise ArtifactError(f"unknown artifact {artifact_id!r}; see list_artifacts")
        async with httpx.AsyncClient(timeout=60, transport=self.transport) as client:
            try:
                response = await client.get(
                    ref.download_url, headers={SERVICE_KEY_HEADER: self.service_key}
                )
            except httpx.HTTPError as exc:
                raise ArtifactError(f"download of {artifact_id} failed") from exc
        if response.status_code != 200:
            raise ArtifactError(f"download of {artifact_id} failed: HTTP {response.status_code}")
        if hashlib.sha256(response.content).hexdigest() != ref.sha256.lower():
            raise ArtifactError(f"artifact {artifact_id} failed its sha256 check")
        try:
            text = response.content.decode("utf-8")
        except UnicodeDecodeError:
            text = f"[binary artifact, {len(response.content)} bytes]"
        self._cache[artifact_id] = text
        return text

    async def json(self, artifact_id: str) -> Any:
        try:
            return json.loads(await self.text(artifact_id))
        except ValueError:
            return None

    async def read(self, artifact_id: str, section: str | None = None, offset: int = 0) -> str:
        """What `read_artifact` returns: an overview, one section, or one page of text."""
        ref = self.refs.get(artifact_id)
        text = await self.text(artifact_id)
        self.read_ids.add(artifact_id)
        doc = await self.json(artifact_id) if text.lstrip().startswith(("{", "[")) else None
        name = ref.schema_id.split("/")[0]
        if name == "i2h.hypotheses" and isinstance(doc, dict):
            return _hypotheses(doc, section)
        if name == "review.findings" and isinstance(doc, dict):
            return _findings(doc)
        if isinstance(doc, dict) and section:
            if section not in doc:
                return f"No key {section!r}. Keys: {', '.join(doc)}"
            return _page(json.dumps(doc[section], indent=1, ensure_ascii=False), offset)
        if doc is not None:
            body = json.dumps(doc, indent=1, ensure_ascii=False)
            head = f"Top-level keys: {', '.join(doc)}\n" if isinstance(doc, dict) else ""
            return head + _page(body, offset)
        return _markdown(text, section, offset)

    # -- facts the guards need --------------------------------------------------------

    async def ids_of(self, artifact_id: str) -> list[str]:
        doc = await self.json(artifact_id)
        if isinstance(doc, dict) and isinstance(doc.get("hypotheses"), list):
            return [str(h.get("id")) for h in doc["hypotheses"] if isinstance(h, dict)]
        return []

    async def latest_review_passed(self, stage_key: str | None) -> bool | None:
        doc = await self._latest_review(stage_key)
        return doc.get("passed") if isinstance(doc, dict) else None

    async def blocked(self, stage_key: str | None) -> tuple[bool, set[str]] | None:
        """Blocking findings of the latest review: (whole artifact blocked, blocked ids).

        A blocking finding anchored to ``hypothesis:<id>`` blocks that id only; any other
        blocking finding blocks the whole artifact. ``None`` when there is no review.
        """
        doc = await self._latest_review(stage_key)
        if not isinstance(doc, dict):
            return None
        whole, ids = False, set()
        for finding in doc.get("findings") or []:
            if not isinstance(finding, dict) or finding.get("severity") != "blocking":
                continue
            locator = str((finding.get("anchor") or {}).get("locator") or "")
            if locator.startswith("hypothesis:"):
                ids.add(locator.removeprefix("hypothesis:").split("/", 1)[0])
            else:
                whole = True
        return whole, ids

    async def _latest_review(self, stage_key: str | None):
        reviews = [
            r
            for r in self.refs.values()
            if r.role == "review"
            and r.schema_id.startswith("review.findings/")
            and (stage_key is None or r.stage_key in (None, stage_key))
        ]
        if not reviews:
            return None
        latest = max(reviews, key=lambda r: r.version)
        return await self.json(latest.artifact_id)


def _page(body: str, offset: int) -> str:
    offset = max(0, offset)
    chunk = body[offset : offset + PAGE]
    rest = len(body) - offset - len(chunk)
    tail = (
        f"\n[... {rest} more characters; read again with offset={offset + PAGE}]"
        if rest > 0
        else ""
    )
    return chunk + tail


def _hypotheses(doc: dict[str, Any], section: str | None) -> str:
    items = [h for h in doc.get("hypotheses", []) if isinstance(h, dict)]
    if section:
        for h in items:
            if str(h.get("id")) == section:
                return json.dumps(h, indent=1, ensure_ascii=False)
        for key in ("held_back", "not_used", "open_tensions", "disagreements"):
            if section == key:
                return _page(json.dumps(doc.get(key), indent=1, ensure_ascii=False), 0)
        return f"No hypothesis {section!r}. Ids: {', '.join(str(h.get('id')) for h in items)}"
    lines = [f"{len(items)} hypotheses (read one with section=<id> for every field):"]
    for h in items:
        lines.append(
            f"- {h.get('id')} [risk={h.get('risk', '?')}, "
            f"evidence={len(h.get('evidence_refs') or [])} cards]: {h.get('statement', '')}"
        )
        if h.get("falsification_criteria"):
            lines.append(f"  falsified if: {h['falsification_criteria']}")
    extra = [k for k in ("held_back", "not_used", "open_tensions", "disagreements") if doc.get(k)]
    if extra:
        lines.append(f"Other sections (read with section=<name>): {', '.join(extra)}")
    return "\n".join(lines)


def _findings(doc: dict[str, Any]) -> str:
    lines = [f"Review round {doc.get('round')}: passed={doc.get('passed')}"]
    for f in doc.get("findings", []):
        anchor = f.get("anchor") or {}
        lines.append(
            f"- {f.get('finding_id')} [{f.get('severity')}/{f.get('category', 'other')}] "
            f"@ {anchor.get('locator')}: {f.get('issue')}"
            + (f" | suggestion: {f.get('suggestion')}" if f.get("suggestion") else "")
        )
    return "\n".join(lines)


def _markdown(text: str, section: str | None, offset: int) -> str:
    headings = list(_HEADING.finditer(text))
    if section:
        wanted = section.strip().lower().lstrip("#").strip()
        for i, match in enumerate(headings):
            if wanted in match.group(2).lower():
                level = len(match.group(1))
                end = len(text)
                for later in headings[i + 1 :]:
                    if len(later.group(1)) <= level:
                        end = later.start()
                        break
                return _page(text[match.start() : end], offset)
        return f"No section matching {section!r}. Sections: " + "; ".join(
            m.group(2) for m in headings
        )
    if offset == 0 and len(text) > PAGE and headings:
        toc = "\n".join(f"{m.group(1)} {m.group(2)}" for m in headings)
        return f"Sections (read one with section=<title>):\n{toc}\n\n---\n" + _page(text, 0)
    return _page(text, offset)
