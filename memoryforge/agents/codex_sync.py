"""Read Codex CLI configuration without exposing secrets in MemoryForge config."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python < 3.11
    import tomli as tomllib  # type: ignore[import-not-found, no-redef]


@dataclass(frozen=True)
class CodexDefaults:
    provider: str | None
    model: str | None
    base_url: str | None
    wire_api: str | None
    api_key: str | None


def find_memoryforge_config(project_root: str | Path | None = None) -> Path | None:
    start = Path(project_root).expanduser().resolve() if project_root else Path.cwd().resolve()
    candidates = [start, *start.parents]
    for root in candidates:
        path = root / ".memoryforge" / "config.json"
        if path.exists():
            return path
    return None


def load_memoryforge_config(project_root: str | Path | None = None) -> dict[str, Any]:
    path = find_memoryforge_config(project_root)
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_memoryforge_config(path: str | Path, config: dict[str, Any]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_codex_defaults(project_root: str | Path | None = None) -> CodexDefaults:
    config = _merged_codex_config(project_root)
    provider = _as_str(config.get("model_provider")) or "OpenAI"
    providers_value = config.get("model_providers")
    providers = providers_value if isinstance(providers_value, dict) else {}
    provider_value = providers.get(provider)
    provider_config = provider_value if isinstance(provider_value, dict) else {}
    return CodexDefaults(
        provider=provider,
        model=_as_str(config.get("model")),
        base_url=_as_str(provider_config.get("base_url")),
        wire_api=_as_str(provider_config.get("wire_api")),
        api_key=_codex_api_key(provider_config),
    )


def project_subagent_config(project_root: str | Path | None = None) -> dict[str, Any]:
    config = load_memoryforge_config(project_root)
    subagent = config.get("subagent")
    return subagent if isinstance(subagent, dict) else {}


def _merged_codex_config(project_root: str | Path | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in _codex_config_paths(project_root):
        data = _load_toml(path)
        if data:
            _deep_merge(merged, data)
    return merged


def _codex_config_paths(project_root: str | Path | None) -> list[Path]:
    paths = [Path.home() / ".codex" / "config.toml"]
    if project_root:
        paths.append(Path(project_root).expanduser().resolve() / ".codex" / "config.toml")
    else:
        paths.append(Path.cwd().resolve() / ".codex" / "config.toml")
    return paths


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def _codex_api_key(provider_config: dict[str, Any]) -> str | None:
    for env_name in (
        _as_str(provider_config.get("api_key_env_var")),
        _as_str(provider_config.get("env_key")),
        "OPENAI_API_KEY",
    ):
        if env_name and os.environ.get(env_name):
            return os.environ[env_name]

    auth_path = Path.home() / ".codex" / "auth.json"
    try:
        auth = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(auth, dict):
        return None
    key = auth.get("OPENAI_API_KEY")
    return key if isinstance(key, str) and key else None


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
