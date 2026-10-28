"""Verify Python-only project links, figures, and recorded regression metrics offline."""
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote
import xml.etree.ElementTree as ET

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def verify():
    problems = []
    markdown_paths = [ROOT / "README.md"]
    for directory in ("docs", "task1_eda", "task2_regression"):
        markdown_paths += list((ROOT / directory).rglob("*.md"))
    checked_links = 0
    for path in markdown_paths:
        content = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", content):
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                continue
            relative = unquote(target.split("#")[0])
            if not (path.parent / relative).exists():
                problems.append(f"Broken local link: {path.relative_to(ROOT)} -> {target}")
            checked_links += 1

    notebooks = list(ROOT.rglob("*.ipynb"))
    if notebooks:
        problems.append("Python-only repository contains notebook files: " +
                        ", ".join(str(p.relative_to(ROOT)) for p in notebooks))

    required_python = [
        ROOT / "run_all.py",
        ROOT / "task1_eda/analysis.py",
        ROOT / "task2_regression/train.py",
        ROOT / "task2_regression/predict.py",
        ROOT / "tests/test_projects.py",
    ]
    for path in required_python:
        if not path.is_file():
            problems.append(f"Missing required Python file: {path.relative_to(ROOT)}")
        else:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    figure_paths = []
    for directory in ("task1_eda", "task2_regression"):
        figure_paths.extend((ROOT / directory / "reports/figures").glob("*.png"))
    if len(figure_paths) != 7:
        problems.append(f"Expected 7 report figures, found {len(figure_paths)}")
    for path in figure_paths:
        try:
            with Image.open(path) as img:
                if img.width < 1000 or img.height < 600:
                    problems.append(f"Figure dimensions too small: {path.relative_to(ROOT)}")
                img.verify()
        except OSError as exc:
            problems.append(f"Invalid figure {path.relative_to(ROOT)}: {exc}")

    ET.parse(ROOT / "docs/banner.svg")
    metrics = json.loads((ROOT / "task2_regression/reports/metrics.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for metric in metrics["test_metrics"].values():
        if f"{metric:.3f}" not in readme:
            problems.append(f"README does not contain recorded metric {metric:.3f}")

    if problems:
        raise AssertionError("\n".join(problems))
    print(json.dumps({
        "local_links_checked": checked_links,
        "python_entry_points_compiled": len(required_python),
        "notebook_files_found": 0,
        "figure_files_verified": len(figure_paths),
        "readme_metrics_reconciled": True,
    }, indent=2))


if __name__ == "__main__":
    try:
        verify()
    except (AssertionError, OSError, ValueError) as exc:
        print(f"Deliverable verification failed: {exc}", file=sys.stderr)
        sys.exit(1)
