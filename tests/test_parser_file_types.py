from __future__ import annotations

from pathlib import Path

from app.ingest.parsers import parse_file


def test_parse_file_supports_code_and_markup(tmp_path: Path) -> None:
    samples = {
        ".html": "<html><body>Sample HTML</body></html>",
        ".py": "print('sample python file')\n",
        ".m": "function y = square(x)\n    y = x.^2;\nend\n",
        ".cpp": "#include <iostream>\nint main() { return 0; }\n",
    }

    for suffix, content in samples.items():
        path = tmp_path / f"document{suffix}"
        path.write_text(content, encoding="utf-8")

        parsed = parse_file(path)

        assert content.strip().splitlines()[0] in parsed.text
        assert parsed.metadata["encoding"].lower() == "utf-8"
