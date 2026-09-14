"""Check committed report/notebook structure and local Markdown links offline."""
import base64
import io
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
                problems.append(f"Broken local link: {path.relative_to(ROOT)} → {target}")
            checked_links += 1
    notebook_results = []
    notebooks = [ROOT / "task1_eda/notebooks/penguins_eda.ipynb",
                 ROOT / "task2_regression/notebooks/auto_mpg_regression.ipynb"]
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
        count, figures, ids = 0, 0, set()
        for cell in notebook["cells"]:
            assert cell["id"] not in ids
            ids.add(cell["id"])
            assert isinstance(cell["source"], list)
            if cell["cell_type"] == "code":
                count += 1
                assert cell["execution_count"] == count
                compile("".join(cell["source"]), str(path), "exec")
                for output in cell["outputs"]:
                    assert output["output_type"] != "error"
                    if "image/png" in output.get("data", {}):
                        image_bytes = base64.b64decode(output["data"]["image/png"], validate=True)
                        with Image.open(io.BytesIO(image_bytes)) as img:
                            img.verify()
                        figures += 1
        notebook_results.append({"notebook": str(path.relative_to(ROOT)), "executed_cells": count, "embedded_figures": figures})
    figure_paths = []
    for directory in ("task1_eda", "task2_regression"):
        figure_paths.extend((ROOT / directory / "reports/figures").glob("*.png"))
    assert len(figure_paths) == 7
    for path in figure_paths:
        with Image.open(path) as img:
            assert img.width >= 1000 and img.height >= 600
            img.verify()
    ET.parse(ROOT / "docs/banner.svg")
    metrics = json.loads((ROOT / "task2_regression/reports/metrics.json").read_text())
    readme = (ROOT / "README.md").read_text()
    for metric in metrics["test_metrics"].values():
        assert f"{metric:.3f}" in readme
    if problems:
        raise AssertionError("\n".join(problems))
    print(json.dumps({"local_links_checked": checked_links, "notebooks": notebook_results,
                      "figure_files_verified": len(figure_paths), "readme_metrics_reconciled": True}, indent=2))


if __name__ == "__main__":
    try:
        verify()
    except (AssertionError, OSError, ValueError) as exc:
        print(f"Deliverable verification failed: {exc}", file=sys.stderr)
        sys.exit(1)
