"""File ID extraction and propagation utilities.

LCM's "lossless" guarantee rests on preserving ``file_xxx`` identifiers
across every compaction round.  Even when prose context is discarded, the
pointer to the external file content is never lost.

This module provides:
- :func:`extract_file_ids` — pull all ``file_<hex>`` references from text.
- :func:`append_file_ids_footer` — attach a ``[LCM File IDs: ...]`` footer to
  a summary string when file IDs are present.
- :func:`append_lossless_footers` — attach file IDs and generic source refs.
- :func:`collect_file_ids_from_nodes` — aggregate file IDs from a list of
  ``SummaryNode`` objects (for condensation input propagation).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# Matches LCM file IDs: ``file_`` followed by 6-32 hex characters.
# The pattern is intentionally broad to catch variations across providers.
_FILE_ID_RE = re.compile(r"\bfile_[0-9a-fA-F]{6,32}\b")

# Footer template for the ``[LCM File IDs: ...]`` footer.
_FILE_IDS_FOOTER_TEMPLATE = "\n\n[LCM File IDs: {ids}]"
_FILE_IDS_FOOTER_RE = re.compile(r"\n*\[LCM File IDs:[^\[\]]*\]\s*$", re.MULTILINE)
_SOURCE_REF_RE = re.compile(r"\b(?:rlm_chunk|message|content|file):[A-Za-z0-9_.:/-]+\b")
_SOURCE_REFS_FOOTER_TEMPLATE = "\n\n[LCM Source Refs: {refs}]"
_SOURCE_REFS_FOOTER_RE = re.compile(r"\n*\[LCM Source Refs:[^\[\]]*\]\s*$", re.MULTILINE)
_INTERNAL_SOURCE_REF_PREFIXES = ("content:lcm-compaction-input:",)


def extract_file_ids(text: str) -> list[str]:
    """
    Extract all ``file_<hex>`` identifiers from *text*.

    Deduplicates and preserves first-occurrence order.

    Args:
        text: Raw text that may contain LCM file ID references.

    Returns:
        Ordered, deduplicated list of file ID strings (e.g.
        ``["file_a1b2c3d4", "file_deadbeef12345678"]``).
    """
    seen: set[str] = set()
    result: list[str] = []
    for match in _FILE_ID_RE.finditer(text):
        fid = match.group()
        if fid not in seen:
            seen.add(fid)
            result.append(fid)
    return result


def extract_file_ids_from_messages(
    messages: Sequence[object],
    max_chars_per_message: int = 100_000,
) -> list[str]:
    """
    Extract all file IDs referenced across a list of messages.

    Concatenates all text content from *messages* and deduplicates.

    Args:
        messages: Messages to scan for file ID references.

    Returns:
        Ordered, deduplicated list of file ID strings.
    """
    seen: set[str] = set()
    result: list[str] = []
    for msg in messages:
        text = _message_text(msg)[:max_chars_per_message]
        for fid in extract_file_ids(text):
            if fid not in seen:
                seen.add(fid)
                result.append(fid)
    return result


def collect_file_ids_from_nodes(nodes: Sequence[object]) -> list[str]:
    """
    Aggregate all file IDs already embedded in a list of summary nodes.

    Each node's content may contain a ``[LCM File IDs: ...]`` footer; this
    function extracts IDs from every node and returns a deduplicated union.

    Args:
        nodes: Summary nodes whose content is scanned for file IDs.

    Returns:
        Ordered, deduplicated list of file ID strings.
    """
    seen: set[str] = set()
    result: list[str] = []
    for node in nodes:
        node_file_ids = getattr(node, "file_ids", None) or []
        if isinstance(node_file_ids, str):
            candidate_ids = extract_file_ids(node_file_ids)
        else:
            candidate_ids = [str(file_id) for file_id in node_file_ids]
        candidate_ids.extend(extract_file_ids(str(getattr(node, "content", ""))))
        for fid in candidate_ids:
            if fid not in seen:
                seen.add(fid)
                result.append(fid)
    return result


def extract_source_refs(text: str) -> list[str]:
    """Extract generic source references such as ``rlm_chunk:<id>`` from text."""
    seen: set[str] = set()
    result: list[str] = []
    for match in _SOURCE_REF_RE.finditer(text):
        ref = match.group()
        if ref not in seen:
            seen.add(ref)
            result.append(ref)
    return result


def collect_source_refs_from_nodes(nodes: Sequence[object]) -> list[str]:
    """Aggregate generic source refs from summary nodes and their content."""
    seen: set[str] = set()
    result: list[str] = []
    for node in nodes:
        candidate_refs = [str(ref) for ref in (getattr(node, "source_refs", None) or [])]
        candidate_refs.extend(extract_source_refs(str(getattr(node, "content", ""))))
        for ref in candidate_refs:
            if ref not in seen:
                seen.add(ref)
                result.append(ref)
    return result


def is_internal_source_ref(ref: str) -> bool:
    """Return True for refs used by MemoryForge internals rather than evidence."""

    return any(ref.startswith(prefix) for prefix in _INTERNAL_SOURCE_REF_PREFIXES)


def public_source_refs(refs: Sequence[str]) -> list[str]:
    """Filter internal bookkeeping refs out of public provenance payloads."""

    return [str(ref) for ref in refs if ref and not is_internal_source_ref(str(ref))]


def append_file_ids_footer(text: str, file_ids: Sequence[str]) -> str:
    """
    Append a ``[LCM File IDs: ...]`` footer to *text* when *file_ids* is non-empty.

    If *text* already contains the footer this function is idempotent — it will
    not duplicate the block.  The footer is always placed at the end.

    Args:
        text: The summary text to annotate.
        file_ids: File IDs to include in the footer.

    Returns:
        Annotated text, or the original text unchanged if *file_ids* is empty.
    """
    if not file_ids:
        return text

    ids_str = ", ".join(file_ids)
    footer = _FILE_IDS_FOOTER_TEMPLATE.format(ids=ids_str)

    # Strip any existing footer before appending the authoritative one.
    text = strip_file_ids_footer(text)

    return text + footer


def append_source_refs_footer(text: str, source_refs: Sequence[str]) -> str:
    """Append a ``[LCM Source Refs: ...]`` footer when generic source refs exist."""
    if not source_refs:
        return text
    refs_str = ", ".join(source_refs)
    footer = _SOURCE_REFS_FOOTER_TEMPLATE.format(refs=refs_str)
    text = strip_source_refs_footer(text)
    return text + footer


def append_lossless_footers(
    text: str,
    *,
    file_ids: Sequence[str],
    source_refs: Sequence[str],
) -> str:
    """Append canonical LCM provenance footers without duplicating old footers."""
    stripped = strip_file_ids_footer(strip_source_refs_footer(text))
    with_file_ids = append_file_ids_footer(stripped, file_ids)
    return append_source_refs_footer(with_file_ids, source_refs)


def strip_file_ids_footer(text: str) -> str:
    """Remove a trailing ``[LCM File IDs: ...]`` footer if present."""
    return _FILE_IDS_FOOTER_RE.sub("", text)


def strip_source_refs_footer(text: str) -> str:
    """Remove a trailing ``[LCM Source Refs: ...]`` footer if present."""
    return _SOURCE_REFS_FOOTER_RE.sub("", text)


def validate_file_id_format(file_id: str) -> bool:
    """Return True when *file_id* matches MemoryForge's ``file_<hex>`` format."""
    return bool(_FILE_ID_RE.fullmatch(file_id))


def verify_lossless(input_messages: Sequence[object], output_text: str) -> bool:
    """Verify every input file ID survives in the compacted output text."""
    input_ids = set(extract_file_ids_from_messages(input_messages))
    output_ids = set(extract_file_ids(output_text))
    return input_ids.issubset(output_ids)


def _message_text(message: object) -> str:
    """Extract text from both MemoryForge simple messages and rich LCM messages."""
    if hasattr(message, "content"):
        return str(getattr(message, "content"))
    if hasattr(message, "text"):
        return str(getattr(message, "text"))
    if hasattr(message, "parts"):
        parts = getattr(message, "parts")
        if isinstance(parts, list):
            return "\n".join(
                str(getattr(part, "text", getattr(part, "content", ""))) for part in parts
            )
        return ""
    return ""
