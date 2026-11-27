# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CORE_DIR.parent


def _helper_references(notebook_name: str) -> set[str]:
    notebook = json.loads((CORE_DIR / notebook_name).read_text())
    references: set[str] = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        references.update(
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "helpers"
        )
    return references


def test_module_zero_and_one_import_without_pandas() -> None:
    code = f"""
import sys
sys.path.insert(0, {str(CORE_DIR)!r})
from helpers import module00, module01
assert 'pandas' not in sys.modules
assert 'matplotlib.pyplot' not in sys.modules
assert module00.__all__ == [
    'configure_tutorial',
    'opening_performance_catalog',
    'plot_relaxation_time_perspective',
    'relax_selected_structures',
    'show_structure_explorer',
]
assert module01.__all__ == [
    'configure_tutorial',
    'load_example_molecule',
    'show_molecule',
]
module01.load_example_molecule('Ammonia')
module01.show_molecule(module01.load_example_molecule('Ammonia'))
assert 'pandas' not in sys.modules
assert 'matplotlib.pyplot' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_configure_tutorial_sets_direct_kernel_cache_defaults() -> None:
    variables = (
        "AIMNET_CACHE_DIR",
        "ALCHEMI_D3_PARAM_FILE",
        "HF_HOME",
        "MPLCONFIGDIR",
        "TORCH_HOME",
        "WARP_CACHE_PATH",
    )
    code = f"""
import os
import sys
from pathlib import Path
sys.path.insert(0, {str(CORE_DIR)!r})
from helpers.common import configure_tutorial
configure_tutorial()
expected_root = Path({str(PROJECT_ROOT / ".alchemi-runtime")!r})
d3_file = expected_root / "dftd3" / "dftd3_parameters.pt"
for variable in {variables!r}:
    if variable == "ALCHEMI_D3_PARAM_FILE" and not d3_file.is_file():
        # Named only where the prepared file exists; the lookup beside the
        # Python runtime applies otherwise.
        assert variable not in os.environ
        continue
    assert Path(os.environ[variable]).is_relative_to(expected_root)
"""
    environment = os.environ.copy()
    for variable in variables:
        environment.pop(variable, None)
    subprocess.run([sys.executable, "-c", code], check=True, env=environment)


def test_module_two_exposes_only_its_notebook_interface() -> None:
    sys.path.insert(0, str(CORE_DIR))
    from helpers import module02

    assert set(module02.__all__) == _helper_references(
        "alchemi-core-02-models-and-simulation.ipynb"
    )


def test_module_three_exposes_only_its_notebook_interface() -> None:
    sys.path.insert(0, str(CORE_DIR))
    from helpers import module03

    assert set(module03.__all__) == _helper_references(
        "alchemi-core-03-scale-a-study.ipynb"
    )


def test_module_three_embeds_the_supplied_scheduling_widget() -> None:
    sys.path.insert(0, str(CORE_DIR))
    from helpers import module03

    html = module03.show_scheduling_widget().data

    assert '<iframe title="Study scheduling routes"' in html
    assert "srcdoc=" in html
    assert "ALCHEMI_SCHED" in html
    assert "data-runtime-note" in html
    assert 'role="note"' in html
    assert ".alchemi-sched .as-bar" in html
    assert "margin:10px 0" in html
    assert "TOTAL = 128, ACTIVE = 64, CAP = 16" in html
    assert "__ALCHEMI_" not in html


def test_module_three_storyboard_keeps_two_md_stage_routes_comparable() -> None:
    source = (CORE_DIR / "assets" / "03-campaign-scheduling-widget-v1.html").read_text()
    timing_block = source.split("var TIMING = [", 1)[1].split("];", 1)[0]
    timings = [
        (int(nvt), int(nve))
        for nvt, nve in re.findall(r"\[(\d+),(\d+)\]", timing_block)
    ]

    fixed_batch_steps = max(nvt for nvt, _ in timings) + max(nve for _, nve in timings)
    fused_batch_steps = max(nvt + nve for nvt, nve in timings)
    mean_system_steps = sum(nvt + nve for nvt, nve in timings) / len(timings)
    inflight_speedup = fixed_batch_steps / mean_system_steps

    assert fixed_batch_steps == fused_batch_steps
    assert 1.45 <= inflight_speedup <= 1.55


def test_configure_tutorial_filters_the_torch_jit_deprecation_notice() -> None:
    import warnings

    from helpers import common

    with warnings.catch_warnings():
        common._ignore_torch_jit_deprecation()
        rule = warnings.filters[0]
        assert rule[0] == "ignore" and rule[2] is DeprecationWarning
        assert rule[1].match("`torch.jit.load` is deprecated. Please switch to `torch.export`.")
        assert rule[3].match("torch.jit._serialization")
        assert not rule[1].match("`torch.compile` is deprecated")


def test_mace_load_guard_drops_the_torch_jit_notice_behind_a_once_rule(monkeypatch) -> None:
    """cuEquivariance runs ``warnings.simplefilter("once")`` at import, in front of the course
    filters; the guard still drops torch's jit notice and shows every other warning."""

    import warnings

    from helpers.model_support import suppress_known_mace_load_noise

    notice = "`torch.jit.load` is deprecated. Please switch to `torch.export`."
    own = notice.replace("load", "script_method")
    jit_file = "/venv/lib/python3.12/site-packages/torch/jit/_serialization.py"
    shown: list[tuple[str, type, str]] = []
    monkeypatch.setattr(
        warnings,
        "showwarning",
        lambda message, category, filename, lineno, file=None, line=None: shown.append(
            (str(message), category, filename)
        ),
    )
    with warnings.catch_warnings():
        with suppress_known_mace_load_noise():
            warnings.simplefilter("once")  # what the cuEquivariance import does
            warnings.warn_explicit(notice, DeprecationWarning, jit_file, 176)
            warnings.warn_explicit(notice.replace("load", "script"), DeprecationWarning, jit_file, 1488)
            # a "once" rule shows a text once, so the learner's own call uses a third text
            warnings.warn_explicit(own, DeprecationWarning, "/my/code.py", 3)
            warnings.warn_explicit("something the learner should read", UserWarning, "/lib/other.py", 5)
    assert shown == [
        (own, DeprecationWarning, "/my/code.py"),
        ("something the learner should read", UserWarning, "/lib/other.py"),
    ]


def test_mace_load_guard_shows_recorded_warnings_when_the_load_fails(monkeypatch) -> None:
    import warnings

    import pytest
    from helpers.model_support import suppress_known_mace_load_noise

    shown: list[str] = []
    monkeypatch.setattr(
        warnings,
        "showwarning",
        lambda message, category, filename, lineno, file=None, line=None: shown.append(str(message)),
    )
    with warnings.catch_warnings(), pytest.raises(RuntimeError):
        with suppress_known_mace_load_noise():
            warnings.simplefilter("always")
            warnings.warn_explicit("checkpoint looks odd", UserWarning, "/lib/other.py", 5)
            raise RuntimeError("load failed")
    assert shown == ["checkpoint looks odd"]
