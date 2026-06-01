"""Rewrite the provided PPTX template with the PS40 slide content."""

from __future__ import annotations

import json
import re
import zipfile
from html import escape
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE = BASE_DIR.parent / "Project_Submission_Template.pptx"
CONTENT = BASE_DIR / "presentation_content.json"
OUTPUT = BASE_DIR / "PS40_Network_Intrusion_Detection_Project.pptx"


def load_content() -> dict[int, list[str]]:
    payload = json.loads(CONTENT.read_text(encoding="utf-8"))
    return {int(item["slide"]): item["lines"] for item in payload["slides"]}


def rewrite_slide(xml_text: str, lines: list[str]) -> str:
    tokens = list(re.findall(r"<a:t>.*?</a:t>", xml_text, flags=re.DOTALL))
    if not tokens:
        return xml_text
    replacements = iter(lines + [""] * max(0, len(tokens) - len(lines)))

    def repl(_: re.Match[str]) -> str:
        return f"<a:t>{escape(next(replacements, ''))}</a:t>"

    return re.sub(r"<a:t>.*?</a:t>", repl, xml_text, flags=re.DOTALL)


def build() -> Path:
    content = load_content()
    with zipfile.ZipFile(TEMPLATE, "r") as source, zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename.startswith("ppt/slides/slide") and info.filename.endswith(".xml"):
                match = re.search(r"slide(\d+)\.xml$", info.filename)
                if match:
                    slide_number = int(match.group(1))
                    if slide_number in content:
                        data = rewrite_slide(data.decode("utf-8", errors="ignore"), content[slide_number]).encode("utf-8")
            target.writestr(info, data)
    return OUTPUT


if __name__ == "__main__":
    print(f"Created {build()}")
