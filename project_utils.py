"""Small shared helpers for reproducible reports and consistent figures."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
NAVY = "#17324d"
TEAL = "#008b8b"
ORANGE = "#e79b3e"
COLORS = [TEAL, ORANGE, "#6574ba"]


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.labelsize": 11, "axes.labelcolor": NAVY,
        "text.color": NAVY, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white",
        "axes.facecolor": "#f7f9fb", "savefig.facecolor": "white",
        "axes.grid": True, "grid.alpha": .18, "grid.linewidth": .7,
    })


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def markdown_table(frame, decimals=3):
    """Render a small frame without requiring the optional tabulate package."""
    frame = frame.copy()
    def fmt(x):
        if pd.isna(x):
            return "NA"
        if isinstance(x, float):
            return f"{x:,.{decimals}f}"
        return str(x).replace("|", "\\|").replace("\n", " ")
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows.extend("| " + " | ".join(fmt(x) for x in row) + " |"
                for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)
