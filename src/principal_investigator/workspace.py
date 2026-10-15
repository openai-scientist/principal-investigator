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

    async def current_ids(self, stage_key: str | None) -> list[str]:
        """Item ids of the latest primary artifact of the current attempt, in order."""
        latest = self._latest([r for r in self.refs.values() if r.role == "primary"], stage_key)
        return list(dict.fromkeys(await self.ids_of(latest.artifact_id))) if latest else []

    async def primary_text(self, stage_key: str | None) -> str | None:
        """Text of the latest primary artifact of the current attempt; None without one."""
        latest = self._latest([r for r in self.refs.values() if r.role == "primary"], stage_key)
        return await self.text(latest.artifact_id) if latest else None

    async def latest_review_passed(self, stage_key: str | None) -> bool | None:
        doc = await self._latest_review(stage_key)
        return doc.get("passed") if isinstance(doc, dict) else None

    async def review_findings(self, stage_key: str | None) -> list[dict]:
        """Findings of the latest review of the current attempt; empty without a review."""
        doc = await self._latest_review(stage_key)
        findings = doc.get("findings") if isinstance(doc, dict) else None
        return [f for f in findings or [] if isinstance(f, dict) and f.get("finding_id")]

    async def blocking(self, stage_key: str | None) -> tuple[bool, dict[str, set[str]]] | None:
        """Blocking findings of the latest review: (whole artifact blocked, id -> finding ids).

        A blocking finding anchored to ``hypothesis:<id>`` blocks that id only; any other
        blocking finding of kind ``fix`` blocks the whole artifact (one that needs evidence or
        a tool travels as a known risk: revising cannot fix it). ``None`` when there is no review.
        """
        doc = await self._latest_review(stage_key)
        if not isinstance(doc, dict):
            return None
        whole, ids = False, {}
        for finding in doc.get("findings") or []:
            if not isinstance(finding, dict) or finding.get("severity") != "blocking":
                continue
            locator = str((finding.get("anchor") or {}).get("locator") or "")
            if locator.startswith("hypothesis:"):
                hypothesis = locator.removeprefix("hypothesis:").split("/", 1)[0]
                ids.setdefault(hypothesis, set())
                if finding.get("finding_id"):
                    ids[hypothesis].add(str(finding["finding_id"]))
            elif finding.get("kind", "fix") == "fix":  # an older engine sends no kind
                whole = True
        return whole, ids

    async def blocked(self, stage_key: str | None) -> tuple[bool, set[str]] | None:
        """Like `blocking`, without the finding ids."""
        found = await self.blocking(stage_key)
        return None if found is None else (found[0], set(found[1]))

    async def review_status(self, stage_key: str | None) -> str | None:
        """One line on what the latest review blocks, for the briefing; None without a review
        or when an artifact cannot be read (the agent can still read them itself)."""
        try:
            blocked = await self.blocked(stage_key)
            if blocked is None:
                return None
            whole, ids = blocked
            if whole:
                return "Latest independent review: a blocking finding on the whole artifact."
            known = await self.current_ids(stage_key)
        except ArtifactError:
            return None
        if not ids:
            return "Latest independent review: no blocking finding."
        clean = [i for i in known if i not in ids]
        return (
            f"Latest independent review: blocking findings on {', '.join(sorted(ids))}; "
            f"without one: {', '.join(clean) or 'none'}."
        )

    async def _latest_review(self, stage_key: str | None):
        latest = self._latest(
            [
                r
                for r in self.refs.values()
                if r.role == "review" and r.schema_id.startswith("review.findings/")
            ],
            stage_key,
        )
        return await self.json(latest.artifact_id) if latest else None

    def _latest(self, refs: list[ConsultArtifact], stage_key: str | None):
        """The highest version of the current attempt. Versions restart with each attempt, so an
        earlier attempt's v4 is older than this one's v2; refs without an attempt id are
        compared by version (older Platforms)."""
        refs = [r for r in refs if stage_key is None or r.stage_key in (None, stage_key)]
        if self.current_stage_run and any(r.stage_run_id for r in refs):
            refs = [r for r in refs if r.stage_run_id == self.current_stage_run]
        return max(refs, key=lambda r: r.version) if refs else None


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
