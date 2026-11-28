# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html as html_lib
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from ase.build import molecule

_NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NOTEBOOK_DIR))

import IPython.display as ipy_display
from helpers import batching_figure, common, viewers_three


@pytest.fixture(autouse=True)
def _reset_style():
    yield
    plt.rcdefaults()


def _configure(monkeypatch):
    shown = []
    monkeypatch.setattr(ipy_display, "display", lambda obj: shown.append(obj))
    common.configure_tutorial()
    return shown


def test_configure_applies_the_light_style_and_emits_no_css(monkeypatch) -> None:
    shown = _configure(monkeypatch)
    assert shown == []
    assert common.current_theme() == "light"
    assert matplotlib.colors.to_hex(plt.rcParams["figure.facecolor"]) == "#ffffff"


def test_configure_takes_no_theme_argument() -> None:
    with pytest.raises(TypeError):
        common.configure_tutorial("dark")


def _spec_catalog():
    from dataclasses import dataclass

    from ase import Atoms

    @dataclass(frozen=True)
    class Spec:
        structure: Atoms
        display_name: str
        domain: str
        purpose: str

        def build(self):
            return self.structure.copy()

    return {"water": Spec(molecule("H2O"), "Water", "Chemistry", "Small")}


def _card_options(selection) -> str:
    document = html_lib.unescape(
        re.search(r'srcdoc="([^"]*)"', selection.widget.children[2].children[0].children[0].value).group(1)
    )
    return re.search(r"mountStructure\(document.getElementById\('root'\),.*?,(\{height:[^}]*\})\);", document).group(1)


def test_explorer_is_light(monkeypatch) -> None:
    _configure(monkeypatch)
    selection = viewers_three.show_structure_explorer_three(_spec_catalog())
    assert 'theme:"light"' in _card_options(selection)
    classes = set(selection.widget._dom_classes)
    assert "av-light" in classes and "av-dark" not in classes and "av-auto" not in classes


def test_player_is_light(monkeypatch) -> None:
    _configure(monkeypatch)
    atoms = molecule("H2O")
    atoms.arrays["forces"] = [[1.0, 0, 0]] * 3
    output = viewers_three.show_relaxation_player_three({"Water": [atoms, atoms.copy()]}).data
    document = html_lib.unescape(re.search(r'srcdoc="([^"]*)"', output).group(1))
    assert 'theme:"light"' in document and 'theme:"dark"' not in document


def test_batching_figure_is_light(monkeypatch) -> None:
    relaxation = {"labels": ["a"], "batch_multiplier": 1, "preview": {"device": "CPU"}}
    _configure(monkeypatch)
    figure = batching_figure.build_batching_figure(relaxation)
    assert matplotlib.colors.to_hex(figure.get_facecolor()) == "#ffffff"
    plt.close(figure)


def test_register_course_fonts_loads_shipped_files(monkeypatch, tmp_path) -> None:
    import shutil

    from matplotlib import font_manager

    source = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSerif.ttf"
    (tmp_path / "shared" / "fonts").mkdir(parents=True)
    shutil.copy(source, tmp_path / "shared" / "fonts" / source.name)
    monkeypatch.setattr(common, "_PROJECT_ROOT", tmp_path)
    assert common.register_course_fonts() == ["DejaVu Serif"]
    assert any(f.fname.endswith(str(tmp_path / "shared" / "fonts" / source.name)) for f in font_manager.fontManager.ttflist)


def test_register_course_fonts_without_a_folder_registers_nothing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(common, "_PROJECT_ROOT", tmp_path)
    assert common.register_course_fonts() == []
