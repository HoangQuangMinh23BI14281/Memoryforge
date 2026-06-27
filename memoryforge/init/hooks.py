"""Codex hook ingestion entrypoints."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

PENDING_TTL_SECONDS = 24 * 60 * 60


def handle_hook_event(
    event: str,
    db_path: str,
    agent_id: str,
    project_root: str,
    stdin_text: str,
) -> dict[str, Any]:
    from memoryforge.api import MemoryForge

    if os.environ.get("MEMORYFORGE_SUBAGENT") == "1":
        return {"event": event, "skipped": "subagent"}

    payload = _parse_payload(stdin_text)
    prompt = _extract_prompt(payload, stdin_text)
    session_id = _extract_session_id(payload)

    if event == "session-start":
        from memoryforge.init.autoload import index_project_markdown

        removed = _cleanup_pending(project_root)
        indexed = index_project_markdown(
            db_path=db_path,
            agent_id=agent_id,
            project_root=project_root,
        )
        return {"event": event, "pending_cleaned": removed, "indexed": indexed}

    if event == "user-prompt-submit":
        pending_result: dict[str, Any] = {"event": event}
        if prompt:
            pending_result["pending"] = _store_pending_prompt(
                project_root=project_root,
                agent_id=agent_id,
                session_id=session_id,
                prompt=prompt,
            )
        return pending_result

    if event == "discard-pending":
        return {
            "event": event,
            "pending_discarded": _discard_pending_prompt(project_root, session_id),
        }

    mf = MemoryForge(db_path)
    try:
        result: dict[str, Any] = {"event": event}
        if event == "stop":
            committed = _commit_pending_prompts(
                mf=mf,
                project_root=project_root,
                agent_id=agent_id,
                session_id=session_id,
            )
            if committed["committed"]:
                result["committed"] = committed
                return result

            if prompt:
                result["ingested"] = mf.ingest_prompt(
                    agent_id=agent_id,
                    prompt=prompt,
                    session_id=session_id,
                    project_root=project_root,
                )
            else:
                result["committed"] = committed
            return result

        if prompt:
            result["ingested"] = mf.ingest_prompt(
                agent_id=agent_id,
                prompt=prompt,
                session_id=session_id,
                project_root=project_root,
            )
        return result
    finally:
        mf.close()


def _parse_payload(stdin_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdin_text) if stdin_text.strip() else {}
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_prompt(payload: dict[str, Any], stdin_text: str) -> str:
    for key in ("prompt", "message", "input", "text"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return "" if payload else stdin_text.strip()


def _extract_session_id(payload: dict[str, Any]) -> str | None:
    for key in (
        "session_id",
        "sessionId",
        "conversation_id",
        "conversationId",
        "thread_id",
        "threadId",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return os.environ.get("MEMORYFORGE_SESSION_ID")


def _pending_dir(project_root: str) -> Path:
    return Path(project_root).expanduser().resolve() / ".memoryforge" / "pending"


def _pending_key(session_id: str | None) -> str:
    if not session_id:
        return "default"
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24]
    return f"session-{digest}"


def _pending_path(project_root: str, session_id: str | None) -> Path:
    return _pending_dir(project_root) / f"{_pending_key(session_id)}.json"


def _store_pending_prompt(
    *,
    project_root: str,
    agent_id: str,
    session_id: str | None,
    prompt: str,
) -> dict[str, Any]:
    directory = _pending_dir(project_root)
    directory.mkdir(parents=True, exist_ok=True)
    path = _pending_path(project_root, session_id)
    prompts = [
        {
            "agent_id": agent_id,
            "session_id": session_id,
            "prompt": prompt,
            "created_at": time.time(),
        }
    ]
    payload = {
        "version": 1,
        "agent_id": agent_id,
        "session_id": session_id,
        "updated_at": time.time(),
        "prompts": prompts,
    }
    _write_json_atomic(path, payload)
    return {"session_id": session_id, "count": len(prompts), "path": str(path)}


def _commit_pending_prompts(
    *,
    mf: Any,
    project_root: str,
    agent_id: str,
    session_id: str | None,
) -> dict[str, Any]:
    paths = _candidate_pending_paths(project_root, session_id)
    if not paths:
        return {"committed": 0, "pending_files": 0}

    committed_results: list[dict[str, Any]] = []
    for path in paths:
        pending = _read_pending_file(path)
        prompts = pending.get("prompts")
        if not isinstance(prompts, list):
            path.unlink(missing_ok=True)
            continue
        for item in prompts:
            if not isinstance(item, dict):
                continue
            prompt = item.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                continue
            item_agent_id = item.get("agent_id")
            item_session_id = item.get("session_id")
            committed_results.append(
                mf.ingest_prompt(
                    agent_id=item_agent_id if isinstance(item_agent_id, str) else agent_id,
                    prompt=prompt,
                    session_id=item_session_id if isinstance(item_session_id, str) else session_id,
                    project_root=project_root,
                )
            )
        path.unlink(missing_ok=True)

    return {
        "committed": len(committed_results),
        "pending_files": len(paths),
        "sessions": [
            result.get("session_id")
            for result in committed_results
            if isinstance(result.get("session_id"), str)
        ],
        "turn_ids": [
            turn_id
            for result in committed_results
            for turn_id in result.get("turn_ids", [])
            if isinstance(turn_id, str)
        ],
    }


def _discard_pending_prompt(project_root: str, session_id: str | None) -> dict[str, Any]:
    paths = _candidate_pending_paths(project_root, session_id)
    discarded = 0
    for path in paths:
        if path.exists():
            path.unlink()
            discarded += 1
    return {"files": discarded}


def _candidate_pending_paths(project_root: str, session_id: str | None) -> list[Path]:
    directory = _pending_dir(project_root)
    if not directory.exists():
        return []
    if session_id:
        path = _pending_path(project_root, session_id)
        return [path] if path.exists() else []
    default_path = _pending_path(project_root, None)
    if default_path.exists():
        return [default_path]
    paths = sorted(directory.glob("*.json"))
    return paths if len(paths) == 1 else []


def _cleanup_pending(project_root: str, *, now: float | None = None) -> int:
    directory = _pending_dir(project_root)
    if not directory.exists():
        return 0
    timestamp = time.time() if now is None else now
    removed = 0
    for path in directory.glob("*.json"):
        try:
            if timestamp - path.stat().st_mtime > PENDING_TTL_SECONDS:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def _read_pending_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(".tmp")
    body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    tmp_path.write_text(body, encoding="utf-8")
    try:
        tmp_path.replace(path)
    except PermissionError:
        path.write_text(body, encoding="utf-8")
        try:
            tmp_path.unlink()
        except OSError:
            pass
