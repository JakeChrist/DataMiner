"""Utilities for exporting conversations and snippets to various formats."""

from __future__ import annotations

import datetime as _dt
import html
import json
import textwrap
from pathlib import Path
from typing import Iterable, Sequence

from .conversation_manager import ConversationTurn


class ExportService:
    """Render conversation history or snippets to export-friendly formats."""

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

    def write_text(self, destination: str | Path, content: str) -> Path:
        path = Path(destination)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

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

    def export_snippets_text(
        self, destination: str | Path, snippets: Iterable[dict | str]
    ) -> Path:
        content = self.snippets_to_text(snippets)
        return self.write_text(destination, content)

    # ------------------------------------------------------------------
    def _turn_to_markdown(self, turn: ConversationTurn, index: int) -> list[str]:
        asked = turn.asked_at.isoformat() if turn.asked_at else "—"
        answered = turn.answered_at.isoformat() if turn.answered_at else "—"
        latency = f"{turn.latency_ms} ms" if turn.latency_ms is not None else "—"
        token_json = json.dumps(turn.token_usage or {}, indent=2, sort_keys=True)
        lines = [
            "",
            f"## Turn {index}",
            "",
            f"**Question:** {turn.question.strip() if turn.question else '—'}",
            "",
            "**Answer:**",
            "",
            textwrap.indent((turn.answer or "—").strip(), "> "),
            "",
            f"*Asked:* {asked}  ",
            f"*Answered:* {answered}  ",
            f"*Latency:* {latency}",
            "",
            "### Citations",
        ]
        if turn.citations:
            for idx, citation in enumerate(turn.citations, start=1):
                lines.append(f"{idx}. {self._format_citation_text(citation)}")
        else:
            lines.append("No citations available.")
        reasoning = self._format_reasoning_markdown(turn)
        if reasoning:
            lines.extend(reasoning)
        lines.append("### Token Usage")
        lines.extend(textwrap.indent(token_json or "{}", "    ").splitlines())
        return lines

    def _turn_to_html(self, turn: ConversationTurn, index: int) -> str:
        asked = html.escape(turn.asked_at.isoformat()) if turn.asked_at else "—"
        answered = html.escape(turn.answered_at.isoformat()) if turn.answered_at else "—"
        latency = f"{turn.latency_ms} ms" if turn.latency_ms is not None else "—"
        answer = html.escape(turn.answer or "—").replace("\n", "<br/>")
        question = html.escape(turn.question or "—")
        citations = "".join(
            f"<li>{html.escape(self._format_citation_text(citation))}</li>"
            for citation in turn.citations or []
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

    def _format_reasoning_markdown(self, turn: ConversationTurn) -> list[str]:
        lines: list[str] = []
        if turn.reasoning_bullets:
            lines.extend(["", "### Reasoning", ""])
            for bullet in turn.reasoning_bullets:
                lines.append(f"- {bullet}")
        if turn.plan:
            lines.extend(["", "### Plan", ""])
            for idx, item in enumerate(turn.plan, start=1):
                status = item.status.replace("_", " ") if item.status else "pending"
                lines.append(f"{idx}. {item.description} [{status}]")
        if turn.assumptions or turn.assumption_decision is not None:
            lines.extend(["", "### Assumptions", ""])
            for assumption in turn.assumptions:
                lines.append(f"- {assumption}")
            decision = turn.assumption_decision
            if decision:
                detail = [f"Decision: {decision.mode.title()}"]
                if decision.rationale:
                    detail.append(f"Rationale: {decision.rationale}")
                if decision.clarifying_question:
                    detail.append(f"Follow-up: {decision.clarifying_question}")
                lines.append("- " + " | ".join(detail))
        if turn.self_check is not None:
            self_check = turn.self_check
            lines.extend(["", "### Self-check", ""])
            lines.append("- Status: " + ("Passed" if self_check.passed else "Flagged"))
            for flag in self_check.flags:
                lines.append(f"- Flag: {flag}")
            if self_check.notes:
                lines.append(f"- Notes: {self_check.notes}")
        return lines

    def _format_reasoning_html(self, turn: ConversationTurn) -> str:
        sections: list[str] = []
        if turn.reasoning_bullets:
            bullets = "".join(f"<li>{html.escape(item)}</li>" for item in turn.reasoning_bullets)
            sections.append(
                f"<div class='section'><h3>Reasoning</h3><ul>{bullets}</ul></div>"
            )
        if turn.plan:
            items = "".join(
                f"<li>{html.escape(item.description)} <em>[{html.escape(item.status or 'pending')}]</em></li>"
                for item in turn.plan
            )
            sections.append(f"<div class='section'><h3>Plan</h3><ol>{items}</ol></div>")
        if turn.assumptions or turn.assumption_decision is not None:
            assumptions = "".join(
                f"<li>{html.escape(text)}</li>" for text in turn.assumptions
            )
            extras: list[str] = []
            decision = turn.assumption_decision
            if decision:
                detail = [f"Decision: {decision.mode.title()}"]
                if decision.rationale:
                    detail.append(f"Rationale: {decision.rationale}")
                if decision.clarifying_question:
                    detail.append(f"Follow-up: {decision.clarifying_question}")
                extras.append(" | ".join(html.escape(part) for part in detail))
            if extras:
                assumptions += "".join(f"<li>{extra}</li>" for extra in extras)
            sections.append(f"<div class='section'><h3>Assumptions</h3><ul>{assumptions or '<li>None</li>'}</ul></div>")
        if turn.self_check is not None:
            self_check = turn.self_check
            flags = "".join(f"<li>{html.escape(flag)}</li>" for flag in self_check.flags)
            extra = f"<p>Notes: {html.escape(self_check.notes)}</p>" if self_check.notes else ""
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
    def _format_citation_text(citation: object) -> str:
        if isinstance(citation, str):
            return citation
        if isinstance(citation, dict):
            source = citation.get("source") or citation.get("title") or citation.get("path")
            location: list[str] = []
            page = citation.get("page")
            section = citation.get("section")
            if page is not None:
                location.append(f"page {page}")
            if section:
                location.append(str(section))
            snippet = citation.get("snippet")
            label = source or "Reference"
            tail = f" ({', '.join(location)})" if location else ""
            if snippet:
                snippet_text = ExportService._strip_html(snippet)
                return f"{label}{tail}: {snippet_text}"
            return f"{label}{tail}".strip()
        return str(citation)

    @staticmethod
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

