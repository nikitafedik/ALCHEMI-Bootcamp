"""Expand ``%load`` cells the way JupyterLab does, for unattended notebook runs.

In JupyterLab, running a cell that holds ``%load agentNN_work/step.py`` replaces
the cell's text with ``# %load agentNN_work/step.py`` followed by the file's text;
the learner then runs the cell again. An unattended run (nbconvert, papermill)
cannot do that second step, so the cell runs nothing. This script writes a copy
of the notebook in which every ``%load`` line is already replaced by the file's
text, so that a fresh-kernel run executes the loaded code as the learner would.

Usage (from the repo root or anywhere):

    python notebooks/tests/expand_load.py NOTEBOOK.ipynb SAMPLE_DIR OUT.ipynb

``SAMPLE_DIR`` holds the sample step files by their file names (for example
``relax_build.py``); a ``%load`` path is matched by its file name. The original
notebook is never changed. The script fails if a ``%load`` file is missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def expand(notebook: Path, sample_dir: Path, out: Path) -> list[str]:
    """Write ``out`` with every ``%load`` line expanded; return the files used."""
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    used: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell["source"]
        lines = source.splitlines() if isinstance(source, str) else "".join(source).splitlines()
        new_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("%load "):
                target = stripped[len("%load "):].strip()
                sample = sample_dir / Path(target).name
                if not sample.is_file():
                    raise FileNotFoundError(f"no sample file for '{stripped}': {sample}")
                new_lines.append(f"# {stripped}")
                new_lines.extend(sample.read_text(encoding="utf-8").splitlines())
                used.append(sample.name)
            else:
                new_lines.append(line)
        text = "\n".join(new_lines)
        cell["source"] = [l + "\n" for l in text.split("\n")[:-1]] + [text.split("\n")[-1]]
        cell["outputs"] = []
        cell["execution_count"] = None
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return used


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    used = expand(Path(argv[1]), Path(argv[2]), Path(argv[3]))
    print(f"expanded {len(used)} %load cell(s): {', '.join(used) if used else 'none'} -> {argv[3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
