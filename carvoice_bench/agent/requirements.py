"""Requirement document normalization with stable source anchors."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4
from xml.etree import ElementTree

from carvoice_bench.agent.models import Requirement


class RequirementIngestor:
    """Normalize Markdown, DOCX, and PDF specifications into requirement records."""

    def ingest(self, path: Union[str, Path]) -> List[Requirement]:
        source = Path(path)
        suffix = source.suffix.lower()
        if suffix in {".md", ".markdown", ".txt"}:
            sections = self._read_markdown(source)
        elif suffix == ".docx":
            sections = self._read_docx(source)
        elif suffix == ".pdf":
            sections = self._read_pdf(source)
        else:
            raise ValueError("requirements must be Markdown, TXT, DOCX, or PDF")
        return [
            Requirement(
                id="req-" + uuid4().hex[:12],
                source_path=str(source),
                source_ref=ref,
                title=title,
                text=text,
            )
            for title, text, ref in sections
            if text.strip()
        ]

    def _read_markdown(self, source: Path) -> List[Tuple[str, str, str]]:
        lines = source.read_text(encoding="utf-8").splitlines()
        sections: List[Tuple[str, str, str]] = []
        title = source.stem
        start = 1
        content: List[str] = []
        for number, line in enumerate(lines, start=1):
            heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
            if heading:
                if content:
                    sections.append((title, "\n".join(content).strip(), f"lines:{start}-{number - 1}"))
                title = heading.group(1)
                start = number
                content = []
            else:
                content.append(line)
        if content:
            sections.append((title, "\n".join(content).strip(), f"lines:{start}-{len(lines)}"))
        return sections

    def _read_docx(self, source: Path) -> List[Tuple[str, str, str]]:
        namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        with zipfile.ZipFile(source) as archive:
            xml = archive.read("word/document.xml")
        root = ElementTree.fromstring(xml)
        paragraphs: List[str] = []
        for node in root.iter(namespace + "p"):
            text = "".join(child.text or "" for child in node.iter(namespace + "t")).strip()
            if text:
                paragraphs.append(text)
        return [(source.stem, text, f"paragraph:{index}") for index, text in enumerate(paragraphs, start=1)]

    def _read_pdf(self, source: Path) -> List[Tuple[str, str, str]]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF ingestion requires the optional `pypdf` dependency") from exc
        reader = PdfReader(str(source))
        sections: List[Tuple[str, str, str]] = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                sections.append((f"{source.stem} page {index}", text, f"page:{index}"))
        return sections


def load_rules(path: Optional[Union[str, Path]]) -> Dict[str, Any]:
    if not path:
        return {}
    rule_path = Path(path)
    text = rule_path.read_text(encoding="utf-8")
    if rule_path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML rules require PyYAML") from exc
    return yaml.safe_load(text) or {}
