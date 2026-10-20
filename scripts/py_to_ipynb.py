"""Convert a notebook-style .py (cells separated by '# %% ...' lines) into an .ipynb you can upload to Kaggle.

Comment lines before the first '# %%' become a markdown cell. Inside cells, '# !cmd' becomes the shell line '!cmd'.
Usage: python scripts/py_to_ipynb.py notebooks/04_gnn.py notebooks/04-gnn.ipynb
"""
import json
import re
import sys
from pathlib import Path


def convert(src: str) -> dict:
    head, *chunks = re.split(r"^# %%[^\n]*\n", src, flags=re.MULTILINE)
    intro = "".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in head.splitlines(keepends=True))
    cells = [{"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(keepends=True)}] if intro.strip() else []
    for chunk in chunks:
        code = re.sub(r"^# !", "!", chunk.strip("\n"), flags=re.MULTILINE) + "\n"
        cells.append({"cell_type": "code", "metadata": {"trusted": True}, "source": code.splitlines(keepends=True),
                      "outputs": [], "execution_count": None})
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}


if __name__ == "__main__":
    nb = convert(Path(sys.argv[1]).read_text())
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            compile("".join(line for line in c["source"] if not line.startswith("!")), f"cell {i}", "exec")
    Path(sys.argv[2]).write_text(json.dumps(nb, indent=1))
    print(f"wrote {sys.argv[2]} with {len(nb['cells'])} cells")
