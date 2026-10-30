"""Download the FastAPI docs (MIT licence) at a pinned release and clean them into data/docs.

Pinned so the documents, and so the golden set's expected sources, never move under you.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = "https://github.com/fastapi/fastapi.git"
TAG = "0.162.0"
SECTIONS = ["tutorial", "advanced", "deployment"]
OUT = Path(__file__).parent / "data" / "docs"

INCLUDE = re.compile(r"^\{\*.*\*\}\s*$", re.M)  # {* ../../docs_src/x.py hl[2] *}: code include, not text
ANCHOR = re.compile(r"\s*\{ #[\w-]+ \}")  # "## Origin { #origin }"


def clean(text: str) -> str:
    text = INCLUDE.sub("", ANCHOR.sub("", text))
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        git = ["git", "-C", tmp]
        subprocess.run(
            [
                "git",
                "clone",
                "-q",
                "--depth",
                "1",
                "--branch",
                TAG,
                "--filter=blob:none",
                "--sparse",
                REPO,
                tmp,
            ],
            check=True,
        )
        subprocess.run([*git, "sparse-checkout", "set", "docs/en/docs"], check=True)
        src = Path(tmp) / "docs" / "en" / "docs"
        shutil.rmtree(OUT, ignore_errors=True)
        n = 0
        for section in SECTIONS:
            for f in sorted((src / section).rglob("*.md")):
                dest = OUT / f.relative_to(src)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(clean(f.read_text(encoding="utf-8")), encoding="utf-8")
                n += 1
        shutil.copy(Path(tmp) / "LICENSE", OUT / "LICENSE")
    print(f"{n} documents from FastAPI {TAG} -> {OUT}")


if __name__ == "__main__":
    main()
