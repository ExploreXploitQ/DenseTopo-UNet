from __future__ import annotations

import re
import subprocess
from pathlib import Path

REQUIRED_PUBLIC_FILES = {
    "README.md",
    "MODEL_CARD.md",
    "CITATION.cff",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "Makefile",
    "assets/densetopo-unet-wordmark.svg",
    "assets/architecture.svg",
    "docs/index.md",
    "docs/architecture.md",
    "docs/data-contract.md",
    "docs/configuration.md",
    "docs/usage.md",
    "docs/reproducibility.md",
    "docs/topology-labels.md",
    "docs/limitations.md",
    ".github/workflows/ci.yml",
}


def test_required_public_project_files_exist() -> None:
    missing = sorted(path for path in REQUIRED_PUBLIC_FILES if not Path(path).is_file())
    assert missing == []


def test_readme_states_scientific_and_deployment_boundaries() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "already decompressed" in readme
    assert "No pretrained checkpoint" in readme
    assert "[D, H, W]" in readme
    assert "SPERR" in readme
    assert "not claim validation" in readme


def test_public_text_contains_no_cjk_characters() -> None:
    roots = [Path("README.md"), Path("MODEL_CARD.md"), Path("docs"), Path("src"), Path("scripts")]
    suffixes = {".cff", ".md", ".py", ".svg", ".yaml", ".yml"}
    paths = [
        path
        for root in roots
        for path in ([root] if root.is_file() else root.rglob("*"))
        if path.is_file() and path.suffix in suffixes
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert re.search(r"[\u3400-\u9fff]", text) is None


def test_local_markdown_links_resolve() -> None:
    markdown_files = [Path("README.md"), Path("MODEL_CARD.md"), *Path("docs").glob("*.md")]
    pattern = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
    broken: list[str] = []
    for markdown in markdown_files:
        for target in pattern.findall(markdown.read_text(encoding="utf-8")):
            clean = target.strip("<>").split("#", maxsplit=1)[0]
            if not clean or "://" in clean or clean.startswith("mailto:"):
                continue
            if not (markdown.parent / clean).resolve().exists():
                broken.append(f"{markdown}: {target}")
    assert broken == []


def test_svg_assets_have_accessible_metadata() -> None:
    for path in (Path("assets/densetopo-unet-wordmark.svg"), Path("assets/architecture.svg")):
        text = path.read_text(encoding="utf-8")
        assert "<title" in text
        assert "<desc" in text


def test_git_tracks_no_model_or_raw_volume_artifacts() -> None:
    completed = subprocess.run(["git", "ls-files"], check=True, capture_output=True, text=True)
    prohibited = re.compile(r"\.(?:ckpt|dat|f32|npy|npz|pt|pth|raw)$")
    assert [path for path in completed.stdout.splitlines() if prohibited.search(path)] == []
