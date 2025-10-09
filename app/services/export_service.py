"""Utilities for exporting conversations and snippets to various formats."""

from __future__ import annotations

import datetime as _dt
import html
import json
import logging
import re
import textwrap
from pathlib import Path
from typing import Iterable, Sequence


_CITATION_MARKER_RE = re.compile(r"\[(\d+)\](?!\()")

from .conversation_manager import ConversationTurn
from ..logging import log_call


logger = logging.getLogger(__name__)


class ExportService:
    """Render conversation history or snippets to export-friendly formats."""

    @log_call(logger=logger, include_result=True)
    def conversation_to_markdown(
        self,
        turns: Sequence[ConversationTurn] | Iterable[ConversationTurn],
        *,
        title: str = "Conversation Export",
        metadata: dict | None = None,
    ) -> str:
        entries = list(turns)
        lines: list[str] = [f"# {title}", ""]
        timestamp = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d %H:%M:%SZ")
        lines.append(f"_Generated: {timestamp}_")
        if metadata:
            lines.append("")
            for key, value in metadata.items():
                lines.append(f"* **{key}**: {value}")
        for index, turn in enumerate(entries, start=1):
            lines.extend(self._turn_to_markdown(turn, index))
        return "\n".join(lines).strip() + "\n"

    @log_call(logger=logger, include_result=True)
    def conversation_to_html(
        self,
        turns: Sequence[ConversationTurn] | Iterable[ConversationTurn],
        *,
        title: str = "Conversation Export",
        metadata: dict | None = None,
    ) -> str:
        entries = list(turns)
        timestamp = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d %H:%M:%SZ")
        head = textwrap.dedent(
            f"""
            <!DOCTYPE html>
            <html lang=\"en\">
              <head>
                <meta charset=\"utf-8\" />
                <title>{html.escape(title)}</title>
                <style>
                  body {{ font-family: Arial, sans-serif; margin: 1.5em; }}
                  h1 {{ border-bottom: 1px solid #999; padding-bottom: 0.3em; }}
                  h2 {{ margin-top: 1.4em; }}
                  .meta {{ color: #555; margin-bottom: 1em; }}
                  .section {{ margin-top: 0.8em; }}
                  .section h3 {{ margin-bottom: 0.3em; }}
                  .citations li {{ margin-bottom: 0.2em; }}
                  pre {{ background: #f6f6f6; padding: 0.6em; overflow-x: auto; }}
                </style>
              </head>
              <body>
            """
        ).strip("\n")
        parts: list[str] = [head, f"<h1>{html.escape(title)}</h1>"]
        parts.append(f"<div class='meta'><strong>Generated:</strong> {timestamp}</div>")
        if metadata:
            items = "".join(
                f"<li><strong>{html.escape(str(key))}:</strong> {html.escape(str(value))}</li>"
                for key, value in metadata.items()
            )
            parts.append(f"<ul class='meta'>{items}</ul>")
        for index, turn in enumerate(entries, start=1):
            parts.append(self._turn_to_html(turn, index))
        parts.append("  </body>\n</html>")
        return "\n".join(parts)

    @log_call(logger=logger, include_result=True)
    def snippets_to_text(self, snippets: Iterable[dict | str]) -> str:
        lines: list[str] = []
        for snippet in snippets:
            if isinstance(snippet, str):
                text = snippet
            elif isinstance(snippet, dict):
                label = snippet.get("label") or snippet.get("source") or "Snippet"
                content = snippet.get("snippet_html") or snippet.get("snippet") or ""
                text = f"{label}:\n{self._strip_html(content)}"
                metadata = snippet.get("metadata_text") or snippet.get("metadata")
                if metadata:
                    text += f"\n{metadata}"
            else:
                text = str(snippet)
            lines.append(text.strip())
        return "\n\n".join(line for line in lines if line)

    @log_call(logger=logger, include_result=True)
    def write_text(self, destination: str | Path, content: str) -> Path:
        path = Path(destination)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.info("Wrote export", extra={"destination": str(path), "bytes": path.stat().st_size})
        return path

    @log_call(logger=logger, include_result=True)
    def export_conversation_markdown(
        self,
        destination: str | Path,
        turns: Sequence[ConversationTurn] | Iterable[ConversationTurn],
        *,
        title: str = "Conversation Export",
        metadata: dict | None = None,
    ) -> Path:
        content = self.conversation_to_markdown(turns, title=title, metadata=metadata)
        return self.write_text(destination, content)

    @log_call(logger=logger, include_result=True)
    def export_conversation_html(
        self,
        destination: str | Path,
        turns: Sequence[ConversationTurn] | Iterable[ConversationTurn],
        *,
        title: str = "Conversation Export",
        metadata: dict | None = None,
    ) -> Path:
        content = self.conversation_to_html(turns, title=title, metadata=metadata)
        return self.write_text(destination, content)

    @log_call(logger=logger, include_result=True)
    def export_snippets_text(
        self, destination: str | Path, snippets: Iterable[dict | str]
    ) -> Path:
        content = self.snippets_to_text(snippets)
        return self.write_text(destination, content)

    # ------------------------------------------------------------------
    @log_call(logger=logger, include_result=True)
    def _turn_to_markdown(self, turn: ConversationTurn, index: int) -> list[str]:
        asked = turn.asked_at.isoformat() if turn.asked_at else "—"
        answered = turn.answered_at.isoformat() if turn.answered_at else "—"
        latency = f"{turn.latency_ms} ms" if turn.latency_ms is not None else "—"
        token_json = json.dumps(turn.token_usage or {}, indent=2, sort_keys=True)
        answer_text = (turn.answer or "—").strip()
        answer_text = self._linkify_markdown(answer_text, turn.citations)
        lines = [
            "",
            f"## Turn {index}",
            "",
            f"**Question:** {turn.question.strip() if turn.question else '—'}",
            "",
            "**Answer:**",
            "",
            textwrap.indent(answer_text, "> "),
            "",
            f"*Asked:* {asked}  ",
            f"*Answered:* {answered}  ",
            f"*Latency:* {latency}",
            "",
            "### Citations",
        ]
        if turn.citations:
            for idx, citation in enumerate(turn.citations, start=1):
                text = self._format_citation_text(citation)
                lines.append(f"{idx}. <a id=\"citation-{idx}\"></a>{text}")
        else:
            lines.append("No citations available.")
        reasoning = self._format_reasoning_markdown(turn)
        if reasoning:
            lines.extend(reasoning)
        lines.append("### Token Usage")
        lines.extend(textwrap.indent(token_json or "{}", "    ").splitlines())
        return lines

    @log_call(logger=logger, include_result=True)
    def _turn_to_html(self, turn: ConversationTurn, index: int) -> str:
        asked = html.escape(turn.asked_at.isoformat()) if turn.asked_at else "—"
        answered = html.escape(turn.answered_at.isoformat()) if turn.answered_at else "—"
        latency = f"{turn.latency_ms} ms" if turn.latency_ms is not None else "—"
        answer = self._linkify_html(turn.answer or "—", turn.citations, preserve_breaks=True)
        question = html.escape(turn.question or "—")
        citations = "".join(
            f"<li id='citation-{index}'>{html.escape(self._format_citation_text(citation))}</li>"
            for index, citation in enumerate(turn.citations or [], start=1)
        )
        if not citations:
            citations = "<li>No citations available.</li>"
        reasoning = self._format_reasoning_html(turn)
        token_json = html.escape(json.dumps(turn.token_usage or {}, indent=2, sort_keys=True))
        sections = [
            f"<h2>Turn {index}</h2>",
            f"<div class='section'><h3>Question</h3><p>{question}</p></div>",
            f"<div class='section'><h3>Answer</h3><p>{answer}</p></div>",
            f"<div class='meta'>Asked: {asked} &nbsp;|&nbsp; Answered: {answered} &nbsp;|&nbsp; Latency: {latency}</div>",
            f"<div class='section'><h3>Citations</h3><ul class='citations'>{citations}</ul></div>",
        ]
        if reasoning:
            sections.append(reasoning)
        sections.append(
            f"<div class='section'><h3>Token Usage</h3><pre>{token_json}</pre></div>"
        )
        return "\n".join(sections)

    @log_call(logger=logger, include_result=True)
    def _format_reasoning_markdown(self, turn: ConversationTurn) -> list[str]:
        lines: list[str] = []
        if turn.reasoning_bullets:
            lines.extend(["", "### Reasoning", ""])
            for bullet in turn.reasoning_bullets:
                lines.append(f"- {self._linkify_markdown(bullet, turn.citations)}")
        if turn.plan:
            lines.extend(["", "### Plan", ""])
            for idx, item in enumerate(turn.plan, start=1):
                status = item.status.replace("_", " ") if item.status else "pending"
                description = self._linkify_markdown(item.description, turn.citations)
                lines.append(f"{idx}. {description} [{status}]")
                if item.rationale:
                    rationale_text = self._linkify_markdown(item.rationale, turn.citations)
                    lines.append(f"    ↳ {rationale_text}")
        if turn.assumptions or turn.assumption_decision is not None:
            lines.extend(["", "### Assumptions", ""])
            for assumption in turn.assumptions:
                lines.append(f"- {self._linkify_markdown(assumption, turn.citations)}")
            decision = turn.assumption_decision
            if decision:
                detail = [f"Decision: {decision.mode.title()}"]
                if decision.rationale:
                    detail.append(
                        f"Rationale: {self._linkify_markdown(decision.rationale, turn.citations)}"
                    )
                if decision.clarifying_question:
                    detail.append(
                        f"Follow-up: {self._linkify_markdown(decision.clarifying_question, turn.citations)}"
                    )
                lines.append("- " + " | ".join(detail))
        if turn.self_check is not None:
            self_check = turn.self_check
            lines.extend(["", "### Self-check", ""])
            lines.append("- Status: " + ("Passed" if self_check.passed else "Flagged"))
            for flag in self_check.flags:
                lines.append(f"- Flag: {self._linkify_markdown(flag, turn.citations)}")
            if self_check.notes:
                lines.append(
                    f"- Notes: {self._linkify_markdown(self_check.notes, turn.citations)}"
                )
        return lines

    @log_call(logger=logger, include_result=True)
    def _format_reasoning_html(self, turn: ConversationTurn) -> str:
        sections: list[str] = []
        if turn.reasoning_bullets:
            bullets = "".join(
                f"<li>{self._linkify_html(item, turn.citations)}</li>"
                for item in turn.reasoning_bullets
            )
            sections.append(
                f"<div class='section'><h3>Reasoning</h3><ul>{bullets}</ul></div>"
            )
        if turn.plan:
            items: list[str] = []
            for item in turn.plan:
                description = self._linkify_html(item.description, turn.citations)
                status = html.escape(item.status or "pending")
                rationale_html = ""
                if item.rationale:
                    rationale_html = (
                        f"<div class='plan-rationale'>{self._linkify_html(item.rationale, turn.citations)}</div>"
                    )
                items.append(
                    f"<li>{description} <em>[{status}]</em>{rationale_html}</li>"
                )
            sections.append(
                f"<div class='section'><h3>Plan</h3><ol>{''.join(items)}</ol></div>"
            )
        if turn.assumptions or turn.assumption_decision is not None:
            assumptions = "".join(
                f"<li>{self._linkify_html(text, turn.citations)}</li>" for text in turn.assumptions
            )
            extras: list[str] = []
            decision = turn.assumption_decision
            if decision:
                detail = [f"Decision: {decision.mode.title()}"]
                if decision.rationale:
                    detail.append(
                        f"Rationale: {self._linkify_html(decision.rationale, turn.citations)}"
                    )
                if decision.clarifying_question:
                    detail.append(
                        f"Follow-up: {self._linkify_html(decision.clarifying_question, turn.citations)}"
                    )
                extras.append(" | ".join(self._linkify_html(part, turn.citations) for part in detail))
            if extras:
                assumptions += "".join(f"<li>{extra}</li>" for extra in extras)
            sections.append(f"<div class='section'><h3>Assumptions</h3><ul>{assumptions or '<li>None</li>'}</ul></div>")
        if turn.self_check is not None:
            self_check = turn.self_check
            flags = "".join(
                f"<li>{self._linkify_html(flag, turn.citations)}</li>" for flag in self_check.flags
            )
            extra = (
                f"<p>Notes: {self._linkify_html(self_check.notes, turn.citations, preserve_breaks=True)}</p>"
                if self_check.notes
                else ""
            )
            sections.append(
                "".join(
                    [
                        "<div class='section'><h3>Self-check</h3>",
                        f"<p>Status: {'Passed' if self_check.passed else 'Flagged'}</p>",
                        f"<ul>{flags}</ul>" if flags else "",
                        extra,
                        "</div>",
                    ]
                )
            )
        return "".join(sections)

    @staticmethod
    @log_call(logger=logger, include_result=True)
    def _format_citation_text(citation: object) -> str:
        if isinstance(citation, str):
            return citation.strip()
        if isinstance(citation, dict):
            data = dict(citation)
            nested = data.get("citation")
            if isinstance(nested, dict):
                merged = dict(nested)
                for key, value in data.items():
                    if key not in merged:
                        merged[key] = value
                data = merged

            label = (
                str(
                    data.get("source")
                    or data.get("title")
                    or data.get("document")
                    or data.get("path")
                    or data.get("file_path")
                    or ""
                ).strip()
            )

            def _format_range(
                start: object,
                end: object,
                *,
                singular: str,
                plural: str,
            ) -> str | None:
                if start is not None:
                    start_text = str(start).strip()
                else:
                    start_text = ""
                if end is not None:
                    end_text = str(end).strip()
                else:
                    end_text = ""
                if start_text and end_text:
                    if start_text == end_text:
                        return f"{singular} {start_text}"
                    return f"{plural} {start_text}–{end_text}"
                if start_text:
                    return f"{singular} {start_text}"
                if end_text:
                    return f"{singular} {end_text}"
                return None

            location_parts: list[str] = []

            step_values: list[str] = []
            steps = data.get("steps")
            if isinstance(steps, Iterable) and not isinstance(steps, (str, bytes, dict)):
                for value in steps:
                    text = str(value).strip()
                    if text:
                        step_values.append(text)
            else:
                single_step = data.get("step") or data.get("step_index")
                if single_step is not None:
                    text = str(single_step).strip()
                    if text:
                        step_values.append(text)
            if step_values:
                prefix = "Step" if len(step_values) == 1 else "Steps"
                location_parts.append(f"{prefix} {', '.join(step_values)}")

            page_range = _format_range(
                data.get("page_start") or data.get("page_begin"),
                data.get("page_end") or data.get("page_finish"),
                singular="Page",
                plural="Pages",
            )
            if page_range:
                location_parts.append(page_range)
            else:
                page = data.get("page") or data.get("page_number")
                if page is not None and str(page).strip():
                    location_parts.append(f"Page {str(page).strip()}")

            line_range = _format_range(
                data.get("line_start"),
                data.get("line_end"),
                singular="Line",
                plural="Lines",
            )
            if line_range:
                location_parts.append(line_range)
            else:
                line = data.get("line")
                if line is not None and str(line).strip():
                    location_parts.append(f"Line {str(line).strip()}")

            timestamp_range = _format_range(
                data.get("start_time") or data.get("time_start"),
                data.get("end_time") or data.get("time_end"),
                singular="Timestamp",
                plural="Timestamps",
            )
            if timestamp_range:
                location_parts.append(timestamp_range)
            else:
                timestamp = (
                    data.get("timestamp")
                    or data.get("time")
                    or data.get("timecode")
                    or data.get("locator")
                )
                if timestamp is not None and str(timestamp).strip():
                    location_parts.append(f"Timestamp {str(timestamp).strip()}")

            section = data.get("section") or data.get("heading")
            if section is not None and str(section).strip():
                location_parts.append(str(section).strip())

            location_text = ", ".join(location_parts)

            snippet_text: str | None = None
            snippet_candidates = (
                data.get("snippet_html"),
                data.get("snippet"),
                data.get("highlight"),
                data.get("preview"),
                data.get("text"),
                data.get("content"),
            )
            for candidate in snippet_candidates:
                if candidate:
                    snippet_text = ExportService._strip_html(str(candidate))
                    if snippet_text:
                        break
            if snippet_text:
                snippet_text = snippet_text.strip()

            metadata_text = data.get("metadata_text")
            if metadata_text:
                metadata_text = ExportService._strip_html(str(metadata_text)).strip()
            else:
                metadata = data.get("metadata")
                if isinstance(metadata, dict):
                    pairs: list[str] = []
                    for key, value in metadata.items():
                        if value is None:
                            continue
                        key_text = str(key).strip()
                        value_text = str(value).strip()
                        if key_text and value_text:
                            pairs.append(f"{key_text}: {value_text}")
                    metadata_text = "; ".join(pairs)
                elif isinstance(metadata, Iterable) and not isinstance(metadata, (str, bytes, dict)):
                    metadata_text = ", ".join(str(item).strip() for item in metadata if str(item).strip())
                elif metadata is not None:
                    metadata_text = ExportService._strip_html(str(metadata)).strip()

            parts = [
                label or "Reference",
                location_text,
                metadata_text or "",
            ]
            info = " · ".join(part for part in parts if part)
            if snippet_text:
                return f"{info} — {snippet_text}" if info else snippet_text
            return info or "Reference"
        return str(citation)

    @staticmethod
    def _extract_citation_link(citation: object) -> str | None:
        if isinstance(citation, dict):
            for key in ("url", "href", "link"):
                value = citation.get(key)
                if value:
                    return str(value)
            nested = citation.get("citation")
            if isinstance(nested, dict):
                return ExportService._extract_citation_link(nested)
        return None

    def _linkify_markdown(
        self, text: str | None, citations: Sequence[object] | None
    ) -> str:
        if not text:
            return ""
        citations = tuple(citations or ())

        def _replacement(match: re.Match[str]) -> str:
            index = int(match.group(1))
            href = self._citation_href(index, citations)
            return f"[{index}]({self._escape_markdown_link(href)})"

        return _CITATION_MARKER_RE.sub(_replacement, text)

    def _linkify_html(
        self,
        text: str | None,
        citations: Sequence[object] | None,
        *,
        preserve_breaks: bool = False,
    ) -> str:
        if text is None:
            return ""
        if text == "":
            return ""
        citations = tuple(citations or ())
        parts: list[str] = []
        last = 0
        for match in _CITATION_MARKER_RE.finditer(text):
            parts.append(html.escape(text[last:match.start()]))
            index = int(match.group(1))
            href = html.escape(self._citation_href(index, citations), quote=True)
            parts.append(
                f"<a href=\"{href}\" class='citation-ref'>[{index}]</a>"
            )
            last = match.end()
        parts.append(html.escape(text[last:]))
        result = "".join(parts)
        if preserve_breaks:
            result = result.replace("\n", "<br/>")
        return result

    @staticmethod
    def _citation_href(index: int, citations: Sequence[object]) -> str:
        if 1 <= index <= len(citations):
            link = ExportService._extract_citation_link(citations[index - 1])
            if link:
                return link
        return f"#citation-{index}"

    @staticmethod
    def _escape_markdown_link(target: str) -> str:
        return target.replace("(", r"\(").replace(")", r"\)")

    @staticmethod
    @log_call(logger=logger, include_result=True)
    def _strip_html(value: str | None) -> str:
        if not value:
            return ""
        cleaned = value.replace("<br>", "\n").replace("<br/>", "\n")
        result: list[str] = []
        in_tag = False
        buffer: list[str] = []
        for char in cleaned:
            if char == "<":
                in_tag = True
                if buffer:
                    result.append("".join(buffer))
                    buffer.clear()
                continue
            if char == ">":
                in_tag = False
                continue
            if not in_tag:
                buffer.append(char)
        if buffer:
            result.append("".join(buffer))
        text = html.unescape("".join(result))
        return " ".join(text.split())


__all__ = ["ExportService"]

