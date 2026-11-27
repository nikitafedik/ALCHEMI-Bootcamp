# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the Agent 04 plotters on small synthetic inputs."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

_NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NOTEBOOK_DIR))

import IPython.display as ipy_display
from helpers import agent04
from helpers import agent04_plots as ap

K = ap.KCAL_PER_EV


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _texts(figure) -> list[str]:
    out = [t.get_text() for t in figure.texts]
    for axis in figure.axes:
        out += [axis.get_title(loc=loc) for loc in ("left", "center", "right")]
        out += [axis.get_xlabel(), axis.get_ylabel()]
        out += [t.get_text() for t in axis.texts]
    for legend in figure.legends:
        out += [t.get_text() for t in legend.get_texts()]
    return [t for t in out if t]


def _scan_labels(points_per_pair: int = 2) -> list[dict]:
    labels = []
    for name in ("phenol - N-methylacetamide", "propyne - methyl azide", "ammonia - benzoate"):
        for k in range(points_per_pair):
            labels.append({"system_name": name, "scale": 1.0 + 0.5 * k, "ref_ccsd_t_cbs": -1.0 - k})
    return labels


def _energies(interaction_kcal: np.ndarray) -> torch.Tensor:
    """AB, A, B triples in eV whose interaction is ``interaction_kcal``, shape (3n, 1)."""

    e = np.zeros((interaction_kcal.size, 3))
    e[:, 1], e[:, 2] = -1.0, -2.0
    e[:, 0] = e[:, 1] + e[:, 2] + interaction_kcal / K
    return torch.tensor(e.reshape(-1, 1))


def test_interaction_kcal_tiny_case() -> None:
    e = np.array([1.0, 0.4, 0.5, 2.0, 1.0, 0.9])
    assert ap.interaction_kcal(e) == pytest.approx([0.1 * K, 0.1 * K])
    with pytest.raises(ValueError):
        ap.interaction_kcal(np.zeros(4))


def test_curves_mae_and_order() -> None:
    labels = _scan_labels()
    ref = np.array([l["ref_ccsd_t_cbs"] for l in labels])
    base = _energies(ref + 2.0)
    composed = _energies(ref + np.array([0.5, -0.5] * 3))
    data = ap.interaction_curves_data(labels, composed=composed, base=base)
    assert list(data["lines"]) == ["base", "composed"]
    assert data["mae"]["base"] == pytest.approx(2.0)
    assert data["mae"]["composed"] == pytest.approx(0.5)


def test_curves_d3_line_needs_base_and_no_d3(monkeypatch) -> None:
    labels = _scan_labels()
    ref = np.array([l["ref_ccsd_t_cbs"] for l in labels])
    d3 = np.tile([-0.03, -0.01, -0.01], len(labels))  # -0.01 eV interaction per point
    monkeypatch.setattr(ap, "d3_scan_energy", lambda: d3)
    base = _energies(ref)
    data = ap.interaction_curves_data(labels, base=base, no_d3=_energies(ref + 1.0))
    assert list(data["lines"]) == ["base", "d3", "no_d3"]
    assert data["mae"]["d3"] == pytest.approx(0.01 * K)
    assert "d3" not in ap.interaction_curves_data(labels, base=base)["lines"]


def test_curves_reject_bad_input() -> None:
    labels = _scan_labels()
    with pytest.raises(TypeError):
        ap.interaction_curves_data(labels, d3=np.zeros(18))
    with pytest.raises(TypeError):
        ap.interaction_curves_data(labels, other=np.zeros(18))
    with pytest.raises(ValueError):
        ap.interaction_curves_data(labels, base=np.zeros(9))


def test_curves_figure_text_from_constants(monkeypatch) -> None:
    labels = _scan_labels()
    ref = np.array([l["ref_ccsd_t_cbs"] for l in labels])
    monkeypatch.setattr(ap, "d3_scan_energy", lambda: np.zeros(3 * len(labels)))
    data = ap.interaction_curves_data(labels, base=_energies(ref + 1), no_d3=_energies(ref), composed=_energies(ref))
    figure = ap.build_interaction_figure(data)
    texts = _texts(figure)
    assert [a.get_title(loc="left") for a in figure.axes] == [ap.CURVES_LABELS["panels"][p] for p in ap.PANEL_ORDER]
    assert ap.CURVES_LABELS["ylabel"] in texts and ap.CURVES_LABELS["xlabel"] in texts
    names = ap.CURVES_LABELS["names"]
    for key in ("base", "d3", "no_d3", "composed"):
        assert ap.CURVES_LABELS["line"].format(name=names[key], mae=data["mae"][key]) in texts
    assert ap.CURVES_LABELS["reference"] in texts
    png = ap.figure_png(figure)
    assert png[:4] == b"\x89PNG"
    assert int.from_bytes(png[16:20], "big") == 1840  # 9.2 in at 200 dpi, no tight crop


def test_ncia_mae_and_title() -> None:
    reference = np.array([-1.0, -2.0, -3.0])
    energy = _energies(reference + np.array([1.0, -1.0, 0.0]))
    data = ap.ncia_accuracy_data(energy, reference)
    assert data["mae"] == pytest.approx(2.0 / 3.0)
    figure = ap.build_ncia_figure(data)
    assert figure.axes[0].get_title(loc="left") == ap.NCIA_LABELS["title"].format(mae=2.0 / 3.0)
    assert set(_texts(figure)) == {
        ap.NCIA_LABELS["title"].format(mae=2.0 / 3.0),
        ap.NCIA_LABELS["xlabel"],
        ap.NCIA_LABELS["ylabel"],
    }
    with pytest.raises(ValueError):
        ap.ncia_accuracy_data(energy, np.zeros(4))


def test_ethanol_table_text_from_constants() -> None:
    data = {
        "largest_force": {"network": 4.444, "d3": 4.451, "coulomb": 0.791, "all": 0.781},
        "oh_relaxed": {"network": 1.0446, "all": 0.9577},
    }
    texts = _texts(ap.build_single_molecule_table(data))
    assert set(ap.ETHANOL_LABELS["header"]) <= set(texts)
    assert set(ap.ETHANOL_LABELS["rows"].values()) <= set(texts)
    assert {"4.44", "4.45", "0.79", "0.78", "1.045", "0.958"} <= set(texts)
    assert texts.count(ap.ETHANOL_LABELS["empty"]) == 2


def test_oh_length_of_ethanol() -> None:
    assert 0.9 < ap._oh_length(ap._ethanol()) < 1.0


def test_basin_rule_toy_examples() -> None:
    # Straddles +-180: one basin. Two far points: two more.
    angles = [(-179.0, 170.0), (179.0, -175.0), (175.0, 178.0), (-80.0, 80.0), (-70.0, 90.0), (70.0, -60.0)]
    assert ap.count_basins(angles) == 3
    assert ap.count_basins(angles, {"tolerance_deg": 5.0}) == 6
    assert ap.count_basins(angles, lambda a: len(a)) == 6


def _dipeptide_batch(n: int, seed: int) -> SimpleNamespace:
    rng = np.random.default_rng(seed)
    positions = rng.normal(scale=2.0, size=(n * 22, 3))
    numbers = np.tile(np.r_[[6, 6, 8, 7, 6, 6, 6, 8, 7, 6], np.ones(12, int)], n)
    return SimpleNamespace(
        positions=torch.tensor(positions), atomic_numbers=torch.tensor(numbers), num_graphs=n
    )


@pytest.mark.parametrize("shape", [(4,), (4, 1)])
def test_conformers_data_and_title(shape) -> None:
    start, end = _dipeptide_batch(4, 1), _dipeptide_batch(4, 2)
    energy = torch.tensor([0.3, -0.2, 0.1, 0.0]).reshape(shape)
    data = ap.conformers_data(start, end, energy)
    assert data["start"].shape == (4, 2) and data["end"].shape == (4, 2)
    assert data["lowest"] == 1 and len(data["lowest_atoms"]) == 22
    pos = end.positions.numpy()[22:44]
    assert data["end"][1, 0] == pytest.approx(agent04.dihedral_deg(pos, agent04.PHI_ATOMS))
    assert data["k"] == ap.count_basins(data["end"])
    figure = ap.build_conformers_figure(data)
    texts = _texts(figure)
    assert figure.axes[0].get_title(loc="left") == ap.CONFORMERS_LABELS["title"].format(n_starts=4, k=data["k"])
    assert set(ap.CONFORMERS_LABELS["minima"].values()) <= set(texts)
    assert ap.CONFORMERS_LABELS["source"] in texts
    assert ap.conformers_data(start, end, energy, basin_rule=lambda a: 7)["k"] == 7


def test_plotter_shows_exactly_one_image(monkeypatch) -> None:
    shown = []
    monkeypatch.setattr(ipy_display, "display", lambda obj: shown.append(obj))
    reference = np.array([-1.0, -2.0])
    monkeypatch.setattr(agent04, "load_ncia_references", lambda: reference)
    ap.plot_ncia_accuracy(_energies(reference))
    assert len(shown) == 1
    html = shown[0].data
    assert html.count("<img") == 1 and "max-width:920px" in html


def test_wrapped_line_legend_from_constants() -> None:
    labels = _scan_labels()
    ref = np.array([l["ref_ccsd_t_cbs"] for l in labels])
    data = ap.interaction_curves_data(labels, wrapped=_energies(ref + 0.25), base=_energies(ref + 1.0))
    assert list(data["lines"]) == ["base", "wrapped"]
    texts = _texts(ap.build_interaction_figure(data))
    names = ap.CURVES_LABELS["names"]
    assert ap.CURVES_LABELS["line"].format(name=names["wrapped"], mae=0.25) in texts


def test_figures_follow_the_standard_and_leave_global_style_alone() -> None:
    before = (list(plt.rcParams["figure.figsize"]), plt.rcParams["font.size"], plt.rcParams["savefig.bbox"])
    figure = ap.build_ncia_figure({"reference": np.array([0.0, 1.0]), "predicted": np.array([0.0, 1.0]), "mae": 0.0})
    assert (list(plt.rcParams["figure.figsize"]), plt.rcParams["font.size"], plt.rcParams["savefig.bbox"]) == before
    axis = figure.axes[0]
    assert figure.get_figwidth() == pytest.approx(9.2)
    assert axis.title.get_fontsize() == pytest.approx(10.8)  # 15 px
    assert axis.title.get_fontweight() == "bold"
    assert axis.xaxis.label.get_fontsize() == pytest.approx(9.36)  # 13 px
    assert axis.get_xticklabels()[0].get_fontsize() == pytest.approx(7.92)  # 11 px


def test_table_and_footnote_sizes() -> None:
    data = {
        "largest_force": {"network": 4.4, "d3": 4.4, "coulomb": 0.8, "all": 0.8},
        "oh_relaxed": {"network": 1.04, "all": 0.96},
    }
    texts = ap.build_single_molecule_table(data).axes[0].texts
    sizes = {t.get_text(): t.get_fontsize() for t in texts}
    assert sizes[ap.ETHANOL_LABELS["header"][0]] == pytest.approx(10.8)
    assert sizes[ap.ETHANOL_LABELS["rows"]["network"]] == pytest.approx(9.36)
    start, end = _dipeptide_batch(2, 1), _dipeptide_batch(2, 2)
    figure = ap.build_conformers_figure(ap.conformers_data(start, end, torch.zeros(2)))
    footnote = [t for t in figure.axes[0].texts if t.get_text() == ap.CONFORMERS_LABELS["source"]]
    assert footnote[0].get_fontsize() == pytest.approx(7.92)


def test_agent04_reexports_plotters() -> None:
    for name in ("plot_interaction_curves", "plot_ncia_accuracy", "show_single_molecule_check", "show_conformers"):
        assert name in agent04.__all__
        assert getattr(agent04, name) is getattr(ap, name)


@pytest.mark.parametrize("shape", [(5,), (5, 1)])
def test_ranking_order_and_values(shape) -> None:
    start, end = _dipeptide_batch(5, 1), _dipeptide_batch(5, 2)
    energy = torch.tensor([0.10, -0.20, 0.00, -0.20, 0.05]).reshape(shape)  # eV, a tie at -0.20
    data = ap.conformers_data(start, end, energy)
    assert data["energy"].shape == (5,)
    ranking = data["ranking"]
    assert [r["rank"] for r in ranking] == [1, 2, 3, 4, 5]
    assert [r["index"] for r in ranking] == [1, 3, 2, 4, 0]  # stable sort keeps 1 before 3
    expected = [0.0, 0.0, 0.20 * K, 0.25 * K, 0.30 * K]
    assert [r["energy_above_kcal_mol"] for r in ranking] == pytest.approx(expected)
    assert ranking[0]["phi"] == pytest.approx(data["end"][1, 0])
    assert ranking[0]["psi"] == pytest.approx(data["end"][1, 1])
    for key in ("start", "end", "n_starts", "k", "lowest", "lowest_atoms"):
        assert key in data


def test_ranking_table_text_from_constants() -> None:
    ranking = [
        {"rank": 1, "index": 4, "phi": -84.8, "psi": 82.0, "energy_above_kcal_mol": 0.0},
        {"rank": 2, "index": 0, "phi": -0.4, "psi": 179.6, "energy_above_kcal_mol": 0.3},
    ]
    figure = ap.build_ranking_table({"ranking": ranking})
    axis = figure.axes[0]
    assert axis.get_title(loc="left") == ap.RANKING_LABELS["title"]
    assert axis.title.get_fontsize() == pytest.approx(10.8)
    sizes = {t.get_text(): t.get_fontsize() for t in axis.texts}
    for text in ap.RANKING_LABELS["header"]:
        assert sizes[text] == pytest.approx(10.8)
    assert {"1", "2", "-85", "82", "0", "180", "0.00", "0.30"} <= set(sizes)
    assert sizes["0.00"] == pytest.approx(9.36)
    assert "-0" not in sizes
    assert int.from_bytes(ap.figure_png(figure)[16:20], "big") == 1840


@pytest.mark.parametrize(("switch", "images"), [(True, 2), (False, 1)])
def test_show_ranking_switch(switch, images, monkeypatch) -> None:
    from helpers import viewers

    shown = []
    monkeypatch.setattr(ipy_display, "display", lambda obj: shown.append(obj))
    monkeypatch.setattr(viewers, "show_molecule", lambda atoms: "viewer")
    monkeypatch.setattr(ap, "SHOW_RANKING", switch)
    start, end = _dipeptide_batch(6, 1), _dipeptide_batch(6, 2)
    ap.show_conformers(start, end, torch.arange(6.0))
    html = [obj.data for obj in shown if hasattr(obj, "data")]
    assert len(html) == images
    assert all(h.count("<img") == 1 for h in html)
    assert "viewer" in shown
    if switch:
        assert shown.index("viewer") < len(shown) - 1  # the table follows the viewer


def test_every_picture_has_a_one_sentence_alt_text():
    """Fix pass 27 Nov: alt texts say what each picture shows; they were axis labels."""

    import inspect
    import re as _re

    from helpers import agent04_plots as _plots

    calls = _re.findall(r"_show\(lambda: \w+\(data\), ([^)]+\]?)\)", inspect.getsource(_plots))
    assert len(calls) == 5 and all(call.startswith("ALT_TEXTS[") for call in calls), calls
    labels = {_plots.CURVES_LABELS["ylabel"], _plots.NCIA_LABELS["ylabel"], _plots.CONFORMERS_LABELS["xlabel"],
              _plots.RANKING_LABELS["title"], " | ".join(_plots.ETHANOL_LABELS["header"])}
    for key, text in _plots.ALT_TEXTS.items():
        assert text.endswith(".") and text.count(". ") == 0 and text not in labels, key
        assert "—" not in text and "·" not in text, key
