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
    *,
    source: str = "memoryforge_hook",
    runtime: str = "auto",
) -> dict[str, Any]:
    if os.environ.get("MEMORYFORGE_SUBAGENT") == "1":
        return {"event": event, "skipped": "subagent"}

    canonical_event = _canonical_event(event)
    payload = _parse_payload(stdin_text)
    prompt = _extract_prompt(payload, stdin_text)
    session_id = _extract_session_id(payload)

    if canonical_event == "session-start":
        removed = _cleanup_pending(project_root)
        indexed: dict[str, Any] = {
            "enabled": False,
            "files": 0,
            "chunks": 0,
            "long_term_items": 0,
            "skipped": "hook session-start does not auto-index by default",
        }
        if os.environ.get("MEMORYFORGE_HOOK_AUTO_INDEX") == "1":
            from memoryforge.init.autoload import index_project_markdown

            indexed = index_project_markdown(
                db_path=db_path,
                agent_id=agent_id,
                project_root=project_root,
            )
        return {
            "event": canonical_event,
            "runtime": runtime,
            "pending_cleaned": removed,
            "indexed": indexed,
        }

    if canonical_event == "user-prompt-submit":
        pending_result: dict[str, Any] = {"event": canonical_event}
        if prompt:
            pending_result["pending"] = _store_pending_prompt(
                project_root=project_root,
                agent_id=agent_id,
                session_id=session_id,
                prompt=prompt,
                source=source,
            )
        context = _record_context_snapshot(
            db_path=db_path,
            agent_id=agent_id,
            session_id=session_id,
            trigger=canonical_event,
        )
        if context is not None:
            pending_result["context"] = context
        return pending_result

    if canonical_event == "post-tool-use":
        tool_turns = _extract_tool_turns(payload, source=source)
        result: dict[str, Any] = {"event": canonical_event, "runtime": runtime}
        if tool_turns:
            result["pending"] = _store_pending_turns(
                project_root=project_root,
                agent_id=agent_id,
                session_id=session_id,
                turns=tool_turns,
                source=source,
            )
        else:
            result["skipped"] = "no tool output in payload"
        return result

    if canonical_event == "pre-compact":
        return {
            "event": canonical_event,
            "context": _record_context_snapshot(
                db_path=db_path,
                agent_id=agent_id,
                session_id=session_id,
                trigger=canonical_event,
            ),
        }

    if canonical_event == "discard-pending":
        return {
            "event": canonical_event,
            "pending_discarded": _discard_pending_prompt(project_root, session_id),
        }

    from memoryforge.api import MemoryForge

    mf = MemoryForge(db_path)
    try:
        hook_result: dict[str, Any] = {"event": canonical_event}
        if canonical_event == "post-compact":
            summary = _extract_compact_summary(payload)
            if summary:
                hook_result["committed"] = _store_completed_turns(
                    mf=mf,
                    project_root=project_root,
                    agent_id=agent_id,
                    session_id=session_id,
                    turns=[
                        {
                            "role": "assistant",
                            "content": f"[MemoryForge compact summary]\n{summary}",
                            "metadata": {
                                "source": source,
                                "event": canonical_event,
                                "runtime": runtime,
                            },
                        }
                    ],
                )
            hook_result["context"] = _record_context_snapshot(
                db_path=db_path,
                agent_id=agent_id,
                session_id=session_id,
                trigger=canonical_event,
            )
            return hook_result

        if canonical_event == "stop":
            committed = _commit_pending_prompts(
                mf=mf,
                project_root=project_root,
                agent_id=agent_id,
                session_id=session_id,
                payload=payload,
                source=source,
            )
            if committed["committed"]:
                hook_result["committed"] = committed
                return hook_result

            if prompt:
                hook_result["ingested"] = mf.ingest_prompt(
                    agent_id=agent_id,
                    prompt=prompt,
                    session_id=session_id,
                    project_root=project_root,
                )
            else:
                hook_result["committed"] = committed
            return hook_result

        if prompt:
            hook_result["ingested"] = mf.ingest_prompt(
                agent_id=agent_id,
                prompt=prompt,
                session_id=session_id,
                project_root=project_root,
            )
        return hook_result
    finally:
        mf.close()


def _canonical_event(event: str) -> str:
    key = event.strip().replace("_", "-").lower()
    aliases = {
        "sessionstart": "session-start",
        "session-start": "session-start",
        "userpromptsubmit": "user-prompt-submit",
        "user-prompt-submit": "user-prompt-submit",
        "posttooluse": "post-tool-use",
        "post-tool-use": "post-tool-use",
        "precompact": "pre-compact",
        "pre-compact": "pre-compact",
        "postcompact": "post-compact",
        "post-compact": "post-compact",
        "subagentstop": "subagent-stop",
        "subagent-stop": "subagent-stop",
        "stop": "stop",
        "discard-pending": "discard-pending",
    }
    return aliases.get(key, key)


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
    source: str = "memoryforge_hook",
) -> dict[str, Any]:
    return _store_pending_turns(
        project_root=project_root,
        agent_id=agent_id,
        session_id=session_id,
        turns=[
            {
                "role": "user",
                "content": prompt,
                "metadata": {"source": source, "event": "user-prompt-submit"},
            }
        ],
        source=source,
    )


def _store_pending_turns(
    *,
    project_root: str,
    agent_id: str,
    session_id: str | None,
    turns: list[dict[str, Any]],
    source: str = "memoryforge_hook",
) -> dict[str, Any]:
    directory = _pending_dir(project_root)
    directory.mkdir(parents=True, exist_ok=True)
    path = _pending_path(project_root, session_id)
    existing = _read_pending_file(path)
    existing_turns = existing.get("turns")
    if not isinstance(existing_turns, list):
        existing_turns = []
    existing_prompts = existing.get("prompts")
    if not isinstance(existing_prompts, list):
        existing_prompts = []
    timestamp = time.time()
    normalized_turns = [_normalize_turn(turn, default_source=source) for turn in turns]
    for turn in normalized_turns:
        if turn["role"] == "user" and turn["content"]:
            existing_prompts.append(
                {
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "prompt": turn["content"],
                    "created_at": timestamp,
                }
            )
    existing_turns.extend(normalized_turns)
    payload = {
        "version": 2,
        "agent_id": agent_id,
        "session_id": session_id,
        "updated_at": timestamp,
        "prompts": existing_prompts,
        "turns": existing_turns,
    }
    _write_json_atomic(path, payload)
    return {
        "session_id": session_id,
        "count": len(existing_prompts),
        "turn_count": len(existing_turns),
        "path": str(path),
    }


def _commit_pending_prompts(
    *,
    mf: Any,
    project_root: str,
    agent_id: str,
    session_id: str | None,
    payload: dict[str, Any] | None = None,
    source: str = "memoryforge_hook",
) -> dict[str, Any]:
    paths = _candidate_pending_paths(project_root, session_id)
    stop_payload = payload or {}
    if not paths:
        direct_turns = _completed_turns_from_payload(stop_payload, source=source)
        if not direct_turns:
            return {"committed": 0, "pending_files": 0}
        return _store_completed_turns(
            mf=mf,
            project_root=project_root,
            agent_id=agent_id,
            session_id=session_id,
            turns=direct_turns,
            pending_files=0,
        )

    committed_results: list[dict[str, Any]] = []
    for path in paths:
        pending = _read_pending_file(path)
        turns = _pending_turns(pending)
        if not turns:
            path.unlink(missing_ok=True)
            continue
        turns.extend(_extract_tool_turns(stop_payload, source=source))
        assistant_text = _extract_assistant_text(stop_payload) or _extract_transcript_assistant_text(
            stop_payload
        )
        if assistant_text:
            turns.append(
                {
                    "role": "assistant",
                    "content": assistant_text,
                    "metadata": {"source": source, "event": "stop"},
                }
            )
        item_agent_id = pending.get("agent_id")
        item_session_id = pending.get("session_id")
        committed_results.append(
            _store_completed_turns(
                mf=mf,
                project_root=project_root,
                agent_id=item_agent_id if isinstance(item_agent_id, str) else agent_id,
                session_id=item_session_id if isinstance(item_session_id, str) else session_id,
                turns=turns,
                pending_files=1,
            )
        )
        path.unlink(missing_ok=True)

    return {
        "committed": sum(int(result.get("committed", 0) or 0) for result in committed_results),
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
        "files_ingested": sum(
            int(result.get("files_ingested", 0) or 0) for result in committed_results
        ),
    }


def _store_completed_turns(
    *,
    mf: Any,
    project_root: str,
    agent_id: str,
    session_id: str | None,
    turns: list[dict[str, Any]],
    pending_files: int = 0,
) -> dict[str, Any]:
    normalized = []
    for turn in turns:
        normalized_turn = _normalize_turn(turn)
        if normalized_turn["content"]:
            normalized.append(normalized_turn)
    if not normalized:
        return {"committed": 0, "pending_files": pending_files, "turn_ids": []}
    resolved_session_id = session_id or _session_id_from_turns(normalized) or _new_hook_session_id()
    turn_ids = mf.store_conversation(
        agent_id=agent_id,
        turns=normalized,
        session_id=resolved_session_id,
    )
    file_ingests = _ingest_referenced_files(
        mf=mf,
        agent_id=agent_id,
        turns=normalized,
        project_root=project_root,
    )
    compaction = mf.lcm_compact_if_needed(
        agent_id,
        resolved_session_id,
        defer_soft=True,
    )
    return {
        "committed": len(turn_ids),
        "pending_files": pending_files,
        "session_id": resolved_session_id,
        "turn_ids": turn_ids,
        "files_ingested": len(file_ingests),
        "file_ingests": file_ingests,
        "lcm_compaction": {
            "triggered": compaction.triggered,
            "rounds": compaction.rounds,
            "before_tokens": compaction.before_tokens,
            "after_tokens": compaction.after_tokens,
            "delta_tokens": compaction.delta_tokens,
            "deferred": compaction.deferred,
            "reason": compaction.reason,
            "summary_node_ids": compaction.summary_node_ids,
        },
    }


def _pending_turns(pending: dict[str, Any]) -> list[dict[str, Any]]:
    turns = pending.get("turns")
    if isinstance(turns, list):
        return [_normalize_turn(turn) for turn in turns if isinstance(turn, dict)]
    prompts = pending.get("prompts")
    if not isinstance(prompts, list):
        return []
    result: list[dict[str, Any]] = []
    for item in prompts:
        if not isinstance(item, dict):
            continue
        prompt = item.get("prompt")
        if isinstance(prompt, str) and prompt:
            result.append(
                {
                    "role": "user",
                    "content": prompt,
                    "metadata": {"source": "memoryforge_hook", "event": "user-prompt-submit"},
                }
            )
    return result


def _completed_turns_from_payload(
    payload: dict[str, Any],
    *,
    source: str = "memoryforge_hook",
) -> list[dict[str, Any]]:
    turns = _extract_turn_list(payload)
    if turns:
        return turns
    assistant_text = _extract_assistant_text(payload) or _extract_transcript_assistant_text(payload)
    if assistant_text:
        return [
            {
                "role": "assistant",
                "content": assistant_text,
                "metadata": {"source": source, "event": "stop"},
            }
        ]
    return _extract_tool_turns(payload, source=source)


def _extract_turn_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("memoryforge_turns", "turns", "completed_turn", "completedTurns"):
        value = payload.get(key)
        if isinstance(value, list):
            return [_normalize_turn(item) for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            return [_normalize_turn(value)]
    for key in ("messages", "conversation"):
        value = payload.get(key)
        if isinstance(value, list) and not _has_pending_like_payload(payload):
            return [_normalize_turn(item) for item in value if isinstance(item, dict)]
    return []


def _has_pending_like_payload(payload: dict[str, Any]) -> bool:
    return any(key in payload for key in ("prompt", "input", "text"))


def _extract_tool_turns(
    payload: dict[str, Any],
    *,
    source: str = "memoryforge_hook",
) -> list[dict[str, Any]]:
    candidates: list[Any] = []
    for key in ("tool_outputs", "toolOutputs", "tool_results", "toolResults", "tool_calls", "toolCalls"):
        value = payload.get(key)
        if isinstance(value, list):
            candidates.extend(value)
        elif isinstance(value, dict):
            candidates.append(value)
    if any(key in payload for key in ("tool_name", "toolName", "tool", "name", "output", "result")):
        candidates.append(payload)
    turns: list[dict[str, Any]] = []
    for item in candidates:
        turn = _tool_turn(item, source=source)
        if turn is not None:
            turns.append(turn)
    return turns


def _tool_turn(
    value: Any,
    *,
    source: str = "memoryforge_hook",
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    tool_name = _first_string(value, "tool_name", "toolName", "name")
    tool_value = value.get("tool")
    if not tool_name and isinstance(tool_value, str):
        tool_name = tool_value
    if not tool_name and isinstance(tool_value, dict):
        tool_name = _first_string(tool_value, "name", "tool_name", "toolName")
    tool_name = tool_name or "tool"
    tool_call_id = _first_string(value, "tool_call_id", "toolCallId", "call_id", "id")
    tool_state = _first_string(value, "tool_state", "toolState", "state", "status") or "completed"
    content_parts = [
        _stringify_payload(value.get(key))
        for key in ("output", "result", "stdout", "stderr", "content", "message")
        if value.get(key) is not None
    ]
    content = "\n".join(part for part in content_parts if part).strip()
    if not content:
        arguments = value.get("arguments") if "arguments" in value else value.get("input")
        content = _stringify_payload(arguments).strip()
    if not content:
        return None
    return {
        "role": "assistant",
        "content": f"[Tool {tool_name}]\n{content}",
        "parts": [
            {
                "part_type": "tool",
                "content": content,
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "tool_state": tool_state,
            }
        ],
        "metadata": {"source": source, "event": "post-tool-use"},
    }


def _extract_assistant_text(payload: dict[str, Any]) -> str:
    for key in (
        "assistant_response",
        "assistantResponse",
        "assistant_output",
        "assistantOutput",
        "final_response",
        "finalResponse",
        "answer",
        "response",
        "output",
        "assistant",
    ):
        value = payload.get(key)
        text = _content_text(value)
        if text:
            return text
    messages = payload.get("messages")
    if isinstance(messages, list):
        for item in reversed(messages):
            if isinstance(item, dict) and str(item.get("role", "")).lower() == "assistant":
                text = _content_text(item.get("content") or item.get("message"))
                if text:
                    return text
    return ""


def _extract_transcript_assistant_text(payload: dict[str, Any]) -> str:
    path = _transcript_path(payload)
    if path is None:
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    assistant_messages: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = item.get("message") if isinstance(item, dict) else None
        candidates = [item]
        if isinstance(message, dict):
            candidates.append(message)
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            role = str(candidate.get("role") or "").lower()
            if role != "assistant":
                continue
            content = _content_text(candidate.get("content") or candidate.get("message"))
            if content:
                assistant_messages.append(content)
    return assistant_messages[-1] if assistant_messages else ""


def _transcript_path(payload: dict[str, Any]) -> Path | None:
    for key in ("transcript_path", "transcriptPath", "transcript", "transcript_file"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return Path(value).expanduser()
    return None


def _extract_compact_summary(payload: dict[str, Any]) -> str:
    for key in ("summary", "compact_summary", "compactSummary", "content", "output"):
        text = _content_text(payload.get(key))
        if text:
            return text
    return ""


def _normalize_turn(
    turn: dict[str, Any],
    *,
    default_source: str | None = None,
) -> dict[str, Any]:
    role = str(turn.get("role") or "user").lower()
    if role not in {"user", "assistant", "system", "tool"}:
        role = "user"
    content = _content_text(turn.get("content"))
    if not content:
        content = _content_text(turn.get("message"))
    normalized: dict[str, Any] = {
        "role": role,
        "content": content,
    }
    metadata = turn.get("metadata")
    if isinstance(metadata, dict):
        normalized["metadata"] = dict(metadata)
    elif default_source:
        normalized["metadata"] = {"source": default_source}
    parts = turn.get("parts")
    if isinstance(parts, list):
        normalized_parts = [dict(part) for part in parts if isinstance(part, dict)]
        if normalized_parts:
            normalized["parts"] = normalized_parts
    for key in ("part_type", "tool_name", "tool_call_id", "tool_state", "token_estimate"):
        if key in turn:
            normalized[key] = turn[key]
    return normalized


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "message", "output"):
            text = _content_text(value.get(key))
            if text:
                return text
    if isinstance(value, list):
        chunks = [_content_text(item) for item in value]
        return "\n".join(chunk for chunk in chunks if chunk).strip()
    return ""


def _first_string(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _stringify_payload(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _session_id_from_turns(turns: list[dict[str, Any]]) -> str | None:
    for turn in turns:
        metadata = turn.get("metadata")
        if isinstance(metadata, dict):
            value = metadata.get("session_id")
            if isinstance(value, str) and value:
                return value
    return None


def _new_hook_session_id() -> str:
    return f"codex_{int(time.time() * 1000):016x}_{hashlib.sha256(os.urandom(16)).hexdigest()[:8]}"


def _ingest_referenced_files(
    *,
    mf: Any,
    agent_id: str,
    turns: list[dict[str, Any]],
    project_root: str,
) -> list[dict[str, Any]]:
    ingests: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for turn in turns:
        if turn.get("role") != "user":
            continue
        for file_ref in mf._resolve_prompt_files(str(turn.get("content") or ""), project_root):
            path = str(file_ref["path"])
            if path in seen_paths:
                continue
            seen_paths.add(path)
            ingests.append(
                mf.ingest_file(
                    agent_id=agent_id,
                    path=Path(path),
                    name=str(file_ref["relative_path"]),
                )
            )
    return ingests


def _record_context_snapshot(
    *,
    db_path: str,
    agent_id: str,
    session_id: str | None,
    trigger: str,
) -> dict[str, Any] | None:
    if not session_id:
        return None
    from memoryforge.api import MemoryForge
    from memoryforge.lcm import EventBus

    mf = MemoryForge(db_path)
    try:
        context = mf.lcm_build_context(session_id)
        payload = {
            "trigger": trigger,
            "token_estimate": context.token_estimate,
            "hard_limit": context.budget.hard_limit,
            "soft_limit": context.budget.soft_limit,
            "message_count": len(context.messages),
            "raw_message_ids": context.raw_message_ids,
            "summary_node_ids": context.summary_node_ids,
            "has_summary": context.has_summary,
            "truncated": context.truncated,
        }
        EventBus(db_path).publish(
            "lcm.context.built",
            payload,
            agent_id=agent_id,
            session_id=session_id,
        )
        return payload
    finally:
        mf.close()


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
