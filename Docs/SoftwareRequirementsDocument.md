# LMStudio Document QA Desktop — Requirements (MVP + Advanced RAG & Reasoning)

## Product summary
A local desktop app that lets the user load a large corpus of documents and ask natural-language questions, returning concise answers grounded in the corpus with citations. Runs fully offline using a local LMStudio model, with a PyQt6 GUI and SQLite storage.

---

## Scope & constraints
- Runs entirely on the user’s machine (offline).
- Implemented **only** in **Python** with **SQLite** for storage and a **PyQt6** GUI.
- Uses a **local LMStudio** model for all AI responses.
- Install via a single `requirements.txt`. **No virtual environments.**
- **No cloud calls** or external telemetry.

---

## Functional requirements

### 1) Corpus intake
- Add individual files and whole folders to a corpus.
- Support common text formats (e.g., PDF, DOCX, TXT, MD).
  - Scanned/unreadable PDFs are marked as “needs OCR” if text is not extractable.
- Show import progress and a summary of successes/failures.
- Detect file changes and allow re-scan/update.
- Remove/exclude documents from the corpus without deleting originals.

### 2) Organize & browse
- View corpus by folder, document, and simple user-defined tags.
- Search across documents by keywords.
- See document details (filename, path, size, dates) and a text preview.
- Tag documents and filter by tag.

### 3) Ask & answer
- Accept natural-language questions about the corpus.
- Return a concise answer grounded in the corpus.
- Include **citations** pointing to specific source documents and locations.
- Support follow-up questions that reference prior turns (chat style).
- Let the user choose answer length (brief / normal / detailed).

### 4) Citations & source viewing
- Show cited passages with document name and location (e.g., page/section).
- Clicking a citation opens a source preview focused on that passage.
- Provide an action to open the source file in the system’s default app.

### 5) Results management & export
- Copy answer with citations to clipboard.
- Export a conversation (questions, answers, citations) to **Markdown** or **HTML**.
- Export selected cited snippets to a text file.

### 6) Projects
- Support multiple named “projects,” each with its own corpus and settings.
- Allow quick switching between projects from the GUI.

### 7) Settings
- LMStudio connection settings (host/port/model).
- Basic retrieval/answer preferences (e.g., context size/top-k, answer length preset).
- UI preferences (light/dark theme, font size).
- Backup/restore of app data (settings, tags, chat history).

### 8) Status & control
- Show live progress for long tasks (importing, scanning, analysis).
- Allow pause/cancel for long tasks.
- Provide clear, actionable error messages.

---

## Advanced RAG requirements (no implementation details)

### 9) Retrieval quality & coverage
- Handle **multi-hop** questions that require combining information from multiple documents.
- Provide **diverse evidence** by avoiding duplicate or near-duplicate passages in results.
- Allow **query scoping** by tag/folder/time range/language to narrow context when desired.
- Surface **top-N candidate sources** with scores and short previews for user inspection.
- Identify and **flag conflicting sources** when they disagree on key facts.
- Support common question intents: **fact lookup**, **definition/explanation**, **comparison/contrast**, **timeline synthesis**, **procedure extraction**, **pro/con evidence lists**.

### 10) Evidence & citations
- Every non-obvious claim in answers must be **traceable to at least one citation**.
- When evidence is **insufficient or contradictory**, the answer must state that clearly and summarize the gap.
- Citations include document name and a **specific location** (e.g., page/section/heading).
- Provide a dedicated **Evidence panel** listing the passages actually used in the answer, in answer order.

### 11) Controls & filters
- User can adjust:
  - Number of sources considered and shown.
  - Citation density (fewer vs. more citations).
  - Query scope (tags/folders/time/language).
- User can **include/exclude** specific sources and **re-ask** the question with those choices applied.

### 12) Negative & edge cases
- If no relevant evidence is found, the system explains **why** (e.g., “no matches in selected tags/folder”) and suggests next steps (broaden scope, rescan).
- If a query is ambiguous, the system asks **one concise clarifying question** or proceeds with a clearly stated assumption.

---

## Reasoning, thinking & planning requirements (no implementation details)

### 13) Reasoning modes
- **Standard mode:** concise answers with minimal rationale.
- **Advanced mode:** adds a **brief reasoning summary** (1–5 bullets) explaining how evidence supports the answer (without revealing raw internal chain-of-thought).
- A **Reasoning verbosity** control selects: Minimal / Brief / Extended summaries.

### 14) Planning for complex queries
- For complex or multi-step questions, show a **Plan summary**: the sub-questions to answer and how results will be combined.
- The app indicates which plan steps have been completed and which are pending or skipped.
- The plan can be **collapsed/expanded** by the user.

### 15) Assumptions & checks
- When the system proceeds with assumptions (due to ambiguity), it lists them briefly at the top of the answer.
- A **Self-check summary** flags likely unsupported claims, stale sources, or conflicts.
- For comparison/timeline tasks, the system highlights **criteria** used to compare or order items.

### 16) Audit trail & continuity
- Save per-turn: question, answer, citations used, reasoning summary, plan summary, assumptions, and self-check flags.
- Follow-up questions can **reuse prior evidence** and reference prior assumptions, with the option to revise them.

### 17) Controls
- Toggles in the UI to: show/hide **Reasoning Summary**, **Plan**, **Assumptions**, and **Evidence panel**.
- A **Sources-only** answer mode that restricts output to extracted facts with citations and minimal phrasing.

---

## GUI requirements (PyQt6)

### Layout (at a glance)
- **Question Input:** textbox to type questions.
- **Answer Area:** read-only area showing AI answers with citations.
- **Corpus Selector:** control to choose a specific folder as the active corpus.
- **LMStudio Panel:** visible connection status and a way to verify/repair the connection.
- **Evidence Panel (toggle):** lists the cited passages used in the answer.
- **Reasoning/Plan toggles:** show/hide reasoning summary and plan summary.

### Components & behavior

#### 1) Question Input
- Accepts single-line and multi-line entry (Enter to submit; Shift+Enter newline).
- Visible **Ask** button and **Clear** action.
- Submission is disabled if LMStudio is disconnected or no corpus folder is selected.
- Keeps a history of recent questions (navigable with Up/Down).

#### 2) Answer Area
- Read-only, scrollable; clearly separates turns.
- Renders **citations** as clickable items that open the source preview.
- **Copy** actions: “Copy answer” and “Copy answer + citations.”
- Displays lightweight metadata per turn (timestamp; token/latency if available).
- When enabled, displays **Reasoning Summary**, **Plan summary**, **Assumptions**, and self-check flags above/below the answer (user choice).

#### 3) Corpus Selector
- Control to **choose one folder** as the active corpus.
- Shows current path, document count, and last scan time.
- Actions: **Change Folder**, **Rescan**, (optional) **Exclude Subfolder**, **View Imports Log**.
- Shows progress and errors during scans without blocking the rest of the UI.
- Prevents asking questions when no folder is selected and explains why.

#### 4) LMStudio Panel
- Fields for **Base URL** and **Model name**.
- **Test Connection** button that reports success/failure with a clear message.
- Live **status indicator** (Connected / Disconnected / Error).
- When disconnected: disables Ask and shows reason + next step.

#### 5) Evidence Panel
- Toggle to show/hide.
- Lists the passages actually used in the answer, in order of use, with per-item **jump-to-source**.
- Allows **include/exclude** of specific sources and **re-ask** with those selections.

#### 6) Reasoning & Plan Controls
- Toggles for **Show Reasoning Summary**, **Show Plan**, **Show Assumptions**.
- Slider/preset for **Reasoning verbosity** (Minimal/Brief/Extended).
- **Sources-only** mode toggle.

### Global UI
- **Top bar:** project name, version, Settings, Help.
- **Theme:** light/dark toggle; readable default font size.
- **Keyboard shortcuts:** Enter/Shift+Enter for input; Ctrl+L to focus input; Ctrl+C to copy.
- **Progress:** long tasks show non-blocking progress and can be canceled.
- **Errors:** concise toasts with user actions (e.g., “LMStudio refused connection—check Base URL”).

### Source viewing (from citations)
- Citation click opens a **right-side preview** with doc name, location (page/section), and highlighted passage.
- Button to **Open in system default app** for the file.

### Minimal settings (persisted)
- LMStudio: Base URL, Model name.
- Answer length preset (brief/normal/detailed).
- Reasoning verbosity default; visibility defaults for Evidence/Plan/Assumptions.
- Theme and font size.
- Active project/corpus folder path.

---

## Non-functional requirements

### Privacy & offline
- All processing is local; no external network access required.
- No telemetry or hidden data collection.

### Performance & capacity (targets)
- Smooth handling of **hundreds of documents**.
- Answers for typical questions appear within a few seconds on a medium corpus.
- Advanced RAG and reasoning features should add **limited extra latency** while remaining usable.
- UI remains responsive during long operations (with progress feedback).

### Reliability
- Safe shutdown: partial progress is preserved; no corrupted state.
- Automatic recovery on launch after an unexpected close.

### Usability & accessibility
- Keyboard-friendly interactions.
- Clear, uncluttered layout: corpus (left), chat (center), sources/evidence (right).
- Large click targets and readable defaults.

### Portability
- Windows-first; reasonable support for macOS/Linux where possible.
- Works with a documented system Python version (no venvs).

### Install & updates
- Single `requirements.txt` for dependencies.
- Simple launch instructions (run the Python entry script).
- App shows version and provides a changelog.

### Data ownership
- User can view where the app stores its database and files.
- Provide a “remove this project” action that deletes app-managed data (not originals).

---

## Out of scope (for now)
- Multi-user accounts or permissions.
- Cloud syncing or web access.
- Automated learning/feedback loops beyond basic tagging.
- Fine-grained document annotation/editing inside the app.

---

## Acceptance criteria

### End-to-end scenarios (MVP + Advanced)
1. **Ingest & browse:** User selects a folder with mixed files → sees progress → can browse documents, search, and tag.
2. **Ask & cite:** With a selected corpus and a working LMStudio connection, user asks a question → receives a concise answer with at least one clickable citation that opens the source preview.
3. **Follow-ups:** User asks a follow-up referencing the prior answer → system maintains context and cites correctly.
4. **Export:** User exports the conversation (with citations) to Markdown/HTML and opens it successfully.
5. **Resilience:** App is closed mid-scan/import → relaunch resumes or cleanly rolls back without data loss.

### Advanced RAG scenarios
6. **Multi-hop:** User asks a question requiring information from multiple documents → answer combines evidence from at least two distinct sources with clear citations.
7. **Conflict flagging:** Two sources conflict on a key fact → app surfaces the conflict and summarizes both sides with citations.
8. **Query scoping:** User restricts scope to a tag/folder/time range → results reflect only that scope.
9. **No-evidence case:** No relevant evidence within the selected scope → app responds with a clear “no evidence found” message and suggests broadening scope or rescanning.
10. **Evidence panel use:** User excludes a cited source and re-asks → answer updates and citations reflect the new selection.

### Reasoning & planning scenarios
11. **Reasoning summary:** Advanced mode shows a brief reasoning summary (bulleted) tied to citations; user can hide/show it.
12. **Plan summary:** For a complex query, app shows a plan summary (sub-questions/steps) and marks completed steps.
13. **Assumptions listed:** When ambiguity exists, the app states assumptions at the top of the answer or asks one clarifying question.
14. **Sources-only mode:** When enabled, answer limits itself to extracted facts with citations and minimal phrasing.
15. **Audit trail:** Conversation history includes reasoning/plan summaries, assumptions, and evidence used for each turn.

---

*End of requirements.*
