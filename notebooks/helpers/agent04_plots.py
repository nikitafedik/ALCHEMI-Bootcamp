# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plotters of Agent 04, Made to order.

``plot_interaction_curves``, ``plot_ncia_accuracy``, ``show_single_molecule_check``
and ``show_conformers``. Each takes the names the learner's agent leaves in the
notebook and does the rest.
Every plotter has three layers: a ``*_data`` function that returns the numbers,
a ``build_*`` function that draws the Matplotlib figure in the course figure
standard (``shared/alchemi-figure.mplstyle``, light, 920 px wide), and the
public function that shows it.
"""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Callable, Mapping, Sequence
from html import escape
from io import BytesIO
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from . import common
from .data_records import PROJECT_ROOT

if TYPE_CHECKING:
    import torch
    from ase import Atoms

# ---------------------------------------------------------------------------
# Wording: every text on a figure or a table
# ---------------------------------------------------------------------------
CURVES_LABELS = {
    "ylabel": "E(AB) - E(A) - E(B) [kcal/mol]",
    "xlabel": "Separation [relative to equilibrium]",
    "line": "{name}   MAE {mae:.2f} kcal/mol",
    "reference": "reference",
    "names": {
        "base": "bare MLIP",
        "d3": "bare MLIP with D3",
        "no_d3": "without D3",
        "wrapped": "wrapped",
        "composed": "composed model",
    },
    # Panel titles, keyed by the scan's ``system_name``.
    "panels": {
        "ammonia - benzoate": "ammonia–benzoate",
        "phenol - N-methylacetamide": "phenol–N-methylacetamide",
        "propyne - methyl azide": "propyne–methyl azide",
    },
}
NCIA_LABELS = {
    "title": "MAE {mae:.2f} kcal/mol",
    "xlabel": "Reference interaction energy [kcal/mol]",
    "ylabel": "Predicted interaction energy [kcal/mol]",
}
ETHANOL_LABELS = {
    "header": ("Ethanol", "Largest force [eV/Å]", "O-H after relaxing [Å]"),
    "rows": {
        "network": "bare MLIP",
        "d3": "with D3",
        "coulomb": "with electrostatics",
        "all": "with both",
    },
    "empty": "–",
}
CONFORMERS_LABELS = {
    # End points within 25 degrees count as one. They are minima only after a tight relaxation, so the
    # title says "distinct end points".
    "title": "{n_starts} starts, {k} distinct end points",
    "xlabel": "phi [degrees]",
    "ylabel": "psi [degrees]",
    "source": "Mironov et al., J. Comput. Chem. 40, 297 (2019)",
    "minima": {"C7eq": "C7eq", "C5": "C5", "C7ax": "C7ax", "beta2": "beta2"},
}
# Table of the lowest conformers ranked by energy, shown under the conformer viewer when SHOW_RANKING is True.
RANKING_LABELS = {
    "title": "Lowest five conformers by energy",
    "header": ("Rank", "phi [degrees]", "psi [degrees]", "Energy above the lowest [kcal/mol]"),
}
SHOW_RANKING = False  # False: show_conformers shows the map and the viewer; True adds the ranking table
# Alternative texts of the pictures the helper draws: one sentence on what each shows.
ALT_TEXTS = {
    "curves": ("Interaction energy against separation for each molecule pair, one line per model and the "
               "CCSD(T)/CBS reference as open circles, with each model's MAE in the legend."),
    "ncia": ("Predicted against CCSD(T)/CBS interaction energy for each complex of the NCIA set, with the "
             "diagonal of perfect agreement and the MAE in the title."),
    "ethanol": ("Table of the largest force on an atom of ethanol for the bare MLIP, with D3, with electrostatics "
                "and with both, and the O-H length after relaxing."),
    "conformers": ("Ramachandran map of alanine dipeptide: grey circles at the starting phi and psi, green dots "
                   "where each conformer ended after relaxing, open diamonds at the published minima."),
    "ranking": "Table of the five lowest conformers by energy, with their phi, psi and energy above the lowest.",
}

# ---------------------------------------------------------------------------
# Science constants
# ---------------------------------------------------------------------------
KCAL_PER_EV = 23.0605
# Panel order of the scans, by ``system_name``.
PANEL_ORDER = ("ammonia - benzoate", "phenol - N-methylacetamide", "propyne - methyl azide")
# Line order: the keywords the learner may pass, plus "d3" which the helper computes.
CURVE_ORDER = ("base", "d3", "no_d3", "wrapped", "composed")
# Published minima of alanine dipeptide (phi, psi) in degrees, Mironov et al. 2019.
PUBLISHED_MINIMA = {
    "C7eq": (-81.9, 81.6),
    "C5": (-158.4, 151.6),
    "C7ax": (74.3, -57.4),
    "beta2": (-90.6, -7.8),
}
# Label offsets from each published minimum, in degrees.
_MINIMA_OFFSETS = {"C7eq": (9, 9), "C5": (9, -16), "C7ax": (9, -16), "beta2": (9, -16)}
DEFAULT_BASIN_RULE = {"tolerance_deg": 25.0}
ETHANOL_FMAX = 0.01  # eV/Å, relaxation threshold of the single-molecule check
ETHANOL_MAX_STEPS = 500

# ---------------------------------------------------------------------------
# Figure standard: light only, 9.2 in wide at 100 px per inch (920 px), saved at 200 dpi
# ---------------------------------------------------------------------------
_STYLE = PROJECT_ROOT / "shared" / "alchemi-figure.mplstyle"
_PALETTE = {
    "nv": "#76B900",
    "deep": "#4B7A00",
    "ink": "#1A1A1A",
    "ink2": "#4A4A4A",
    "mute": "#6F6F6F",
    "mid": "#8A8A8A",
    "ref": "#B8B8B8",
    "rule": "#D4D4D4",
    "ring": "#FFFFFF",
}
WIDTH_IN = 9.2
DISPLAY_PX = 920
SAVE_DPI = 200
# Fixed boxes for the square plots, in px of the 920 px canvas: the left axes edge is the
# one constrained layout gives the interaction curves (55 px); top leaves room for the title.
_LEFT_PX = 55
_TOP_PX = 32
_SQUARE_HEIGHT_IN = 6.4
_NCIA_BOTTOM_PX = 52  # tick labels and x label
_MAP_BOTTOM_PX = 76  # tick labels, x label and the source line
_MAP_PAD_DEG = 6  # markers on the +-180 edges stay whole
_MAP_LEFT_SHIFT_PX = 16  # the map's wider tick labels need room for the y label
_TABLE_ROW_PX = 34
_RANKING_ROWS = 5
_RANKING_TITLE_PX = 32  # room for the 15 px bold title above the table
_RANKING_WIDTHS = (0.14, 0.2, 0.2, 0.46)  # column shares of the width; the energy header is long


def _styled(build: Callable[..., Any]) -> Callable[..., Any]:
    """Run a ``build_*`` function inside the course figure style, leaving global settings alone."""

    def wrapper(data: Any) -> Any:
        common.register_course_fonts()  # before the style names the course font
        with plt.style.context(str(_STYLE)):
            return build(data)

    wrapper.__name__, wrapper.__qualname__, wrapper.__doc__ = build.__name__, build.__qualname__, build.__doc__
    return wrapper


def _figure(height_in: float, **kw: Any) -> tuple[Any, Any]:
    return plt.subplots(figsize=(WIDTH_IN, height_in), **kw)


def _square_figure(bottom_px: float, shift_px: float = 0.0) -> tuple[Any, Any]:
    """A 920 px wide figure with one square axes box at the standard's left edge plus ``shift_px``."""

    figure, axis = _figure(_SQUARE_HEIGHT_IN)
    figure.set_layout_engine("none")  # a fixed box keeps the left edge on the standard
    height_px = _SQUARE_HEIGHT_IN * 100
    side = height_px - _TOP_PX - bottom_px
    axis.set_position([(_LEFT_PX + shift_px) / 920, bottom_px / height_px, side / 920, side / height_px])
    return figure, axis


def figure_png(figure: Any) -> bytes:
    """Return the figure as a 2x PNG (1840 px wide), saved in the course figure style."""

    common.register_course_fonts()
    buffer = BytesIO()
    with plt.style.context(str(_STYLE)):  # fonts and layout resolve at save time
        figure.savefig(buffer, format="png", dpi=SAVE_DPI, facecolor=figure.get_facecolor())
    return buffer.getvalue()


def _show(build: Callable[[], Any], alt: str) -> None:
    """Draw the figure and display it as one PNG image, 920 px wide."""

    from IPython.display import HTML, display

    figure = build()
    payload = b64encode(figure_png(figure)).decode("ascii")
    plt.close(figure)
    display(
        HTML(
            f'<img src="data:image/png;base64,{payload}" alt="{escape(alt, quote=True)}" '
            f'style="display:block;box-sizing:border-box;width:100%;max-width:{DISPLAY_PX}px;'
            'height:auto;margin:0">'
        )
    )


def _array(values: Any) -> np.ndarray:
    """Return a tensor or array-like as a float NumPy array on the host."""

    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    return np.asarray(values, dtype=float)


def interaction_kcal(energy: Any) -> np.ndarray:
    """E(AB) - E(A) - E(B) in kcal/mol from energies in eV ordered AB, A, B."""

    e = _array(energy).reshape(-1)
    if e.size % 3:
        raise ValueError(f"energies must come in AB, A, B triples, got {e.size} values")
    return (e[0::3] - e[1::3] - e[2::3]) * KCAL_PER_EV


def mae(predicted: Any, reference: Any) -> float:
    """Mean absolute error of two equal-length arrays."""

    p, r = np.asarray(predicted, dtype=float), np.asarray(reference, dtype=float)
    if p.shape != r.shape:
        raise ValueError(f"shapes differ: {p.shape} and {r.shape}")
    return float(np.abs(p - r).mean())


# ---------------------------------------------------------------------------
# 1. Interaction curves
# ---------------------------------------------------------------------------
_D3_SCAN_CACHE: dict[str, np.ndarray] = {}


def d3_scan_energy() -> np.ndarray:
    """Return Toolkit's D3(BJ) energy in eV of the 90 scan structures, shape (90,).

    Uses ``DFTD3ModelWrapper`` with the settings of the AIMNet2 checkpoint
    (``d3_settings``), on CUDA when available. The result is cached.
    """

    if "energy" not in _D3_SCAN_CACHE:
        import torch
        from nvalchemi.data import AtomicData, Batch
        from nvalchemi.models import DFTD3ModelWrapper

        from .agent04 import d3_settings, load_nci_scans
        from .model_support import prepare_model_batch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        atoms, _ = load_nci_scans()
        d3 = DFTD3ModelWrapper(**d3_settings()).to(device).eval()
        d3.set_config("active_outputs", {"energy"})
        batch = Batch.from_data_list([AtomicData.from_atoms(a) for a in atoms]).to(device)
        batch = prepare_model_batch(batch, d3)
        with torch.no_grad():
            energy = d3(batch)["energy"].detach()
        _D3_SCAN_CACHE["energy"] = _array(energy).reshape(-1)
    return _D3_SCAN_CACHE["energy"]


def interaction_curves_data(scan_labels: Sequence[Mapping[str, Any]], **energies: Any) -> dict[str, Any]:
    """Return interaction energies (kcal/mol) and MAEs for each line of the curves plot.

    ``energies`` are the learner's arrays, shape (90, 1) or (90,) in eV, ordered
    AB, A, B per scan point. With both ``base`` and ``no_d3`` the line "d3"
    (bare MLIP with D3) is added: ``base`` plus Toolkit's D3 term.
    """

    unknown = set(energies) - set(CURVE_ORDER) | ({"d3"} & set(energies))
    if unknown:
        raise TypeError(f"unknown energy keywords {sorted(unknown)}; use {[k for k in CURVE_ORDER if k != 'd3']}")
    if not energies:
        raise TypeError("pass at least one energy, for example base=base_energy")
    n = len(scan_labels)
    reference = np.array([float(l["ref_ccsd_t_cbs"]) for l in scan_labels])
    lines: dict[str, np.ndarray] = {}
    for key, value in energies.items():
        ev = _array(value).reshape(-1)
        if ev.size != 3 * n:
            raise ValueError(f"{key}: expected {3 * n} energies (AB, A, B per scan point), got {ev.size}")
        lines[key] = interaction_kcal(ev)
    if "base" in energies and "no_d3" in energies:
        lines["d3"] = interaction_kcal(_array(energies["base"]).reshape(-1) + d3_scan_energy())
    ordered = {k: lines[k] for k in CURVE_ORDER if k in lines}
    return {
        "pair": [str(l["system_name"]) for l in scan_labels],
        "scale": np.array([float(l["scale"]) for l in scan_labels]),
        "reference": reference,
        "lines": ordered,
        "mae": {k: mae(v, reference) for k, v in ordered.items()},
    }


def _curve_styles(pal: Mapping[str, str]) -> dict[str, tuple[str, Any, float]]:
    return {
        "base": (pal["mid"], "-", 1.8),
        "d3": (pal["ref"], (0, (4, 2)), 1.8),
        "no_d3": (pal["deep"], (0, (1.5, 1.5)), 1.8),
        "wrapped": (pal["ink2"], (0, (6, 2, 1.5, 2)), 1.8),
        "composed": (pal["nv"], "-", 2.2),
    }


@_styled
def build_interaction_figure(data: Mapping[str, Any]) -> Any:
    """Draw the three scan panels, legend underneath."""

    pal = _PALETTE
    styles = _curve_styles(pal)
    pair = np.array(data["pair"])
    panels = [p for p in PANEL_ORDER if p in set(pair)] + sorted(set(pair) - set(PANEL_ORDER))
    figure, axes = _figure(4.4, ncols=len(panels), sharey=True)
    axes = np.atleast_1d(axes)
    names = CURVES_LABELS["names"]
    for axis, name in zip(axes, panels, strict=True):
        m = pair == name
        order = np.argsort(data["scale"][m])
        x = data["scale"][m][order]
        for key, values in data["lines"].items():
            color, style, width = styles[key]
            label = CURVES_LABELS["line"].format(name=names[key], mae=data["mae"][key])
            axis.plot(x, values[m][order], color=color, ls=style, lw=width, label=label)
        axis.plot(
            x,
            data["reference"][m][order],
            "o",
            mfc=pal["ring"],
            mec=pal["ink"],
            mew=1.2,
            ms=4.5,
            zorder=5,
            label=CURVES_LABELS["reference"],
        )
        axis.set_title(CURVES_LABELS["panels"].get(name, name))
        axis.axhline(0, color=pal["rule"], lw=0.8, zorder=0)
    axes[0].set_ylabel(CURVES_LABELS["ylabel"])
    axes[len(axes) // 2].set_xlabel(CURVES_LABELS["xlabel"])
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        columnspacing=1.4,
        handlelength=2.4,
    )
    return figure


def plot_interaction_curves(scan_labels: Sequence[Mapping[str, Any]], **energies: Any) -> None:
    """Plot E(AB) - E(A) - E(B) along the three scans, one line per energy keyword.

    Keywords: ``base``, ``composed``, ``no_d3`` and ``wrapped`` (each (90, 1)
    in eV). The CCSD(T)/CBS references are drawn as open circles and each line's
    MAE is in the legend. With ``base`` and ``no_d3`` the plot also shows the
    bare MLIP with D3, which this helper computes with Toolkit's D3 term.
    """

    data = interaction_curves_data(scan_labels, **energies)
    _show(lambda: build_interaction_figure(data), ALT_TEXTS["curves"])


# ---------------------------------------------------------------------------
# 2. NCIA accuracy
# ---------------------------------------------------------------------------
def ncia_accuracy_data(ncia_energy: Any, reference: Any = None) -> dict[str, Any]:
    """Return predicted and reference interaction energies (kcal/mol) and their MAE."""

    if reference is None:
        from .agent04 import load_ncia_references

        reference = load_ncia_references()
    reference = np.asarray(reference, dtype=float).reshape(-1)
    predicted = interaction_kcal(ncia_energy)
    if predicted.size != reference.size:
        raise ValueError(
            f"expected {3 * reference.size} energies (AB, A, B per complex), got {3 * predicted.size}"
        )
    return {"predicted": predicted, "reference": reference, "mae": mae(predicted, reference)}


@_styled
def build_ncia_figure(data: Mapping[str, Any]) -> Any:
    """Draw predicted against reference interaction energy, square axes centred in the canvas."""

    pal = _PALETTE
    ref, pred = data["reference"], data["predicted"]
    figure, axis = _square_figure(_NCIA_BOTTOM_PX)
    lo, hi = min(ref.min(), pred.min()) - 1, max(ref.max(), pred.max()) + 1
    axis.plot([lo, hi], [lo, hi], color=pal["ref"], lw=1.0, zorder=1)
    axis.plot(ref, pred, "o", ms=4, mfc=pal["nv"], mec=pal["ring"], mew=0.4, alpha=0.85, zorder=3)
    axis.set_xlim(lo, hi)
    axis.set_ylim(lo, hi)
    axis.set_aspect("equal")
    axis.grid(axis="x")
    axis.set_title(NCIA_LABELS["title"].format(mae=data["mae"]))
    axis.set_xlabel(NCIA_LABELS["xlabel"])
    axis.set_ylabel(NCIA_LABELS["ylabel"])
    return figure


def plot_ncia_accuracy(ncia_energy: Any) -> None:
    """Plot predicted against CCSD(T)/CBS interaction energy for the 205 NCIA complexes.

    ``ncia_energy`` has shape (615, 1) in eV, AB, A, B per complex in loader order.
    """

    data = ncia_accuracy_data(ncia_energy)
    _show(lambda: build_ncia_figure(data), ALT_TEXTS["ncia"])


# ---------------------------------------------------------------------------
# 3. Single-molecule check
# ---------------------------------------------------------------------------
def _ethanol() -> Atoms:
    from ase.build import molecule

    atoms = molecule("CH3CH2OH")
    atoms.info["charge"] = 0
    return atoms


def _oh_length(atoms: Atoms) -> float:
    symbols = atoms.get_chemical_symbols()
    o = symbols.index("O")
    d = atoms.get_all_distances()[o]
    return float(min(d[i] for i, s in enumerate(symbols) if s == "H"))


def _ethanol_batch(atoms: Atoms, device: torch.device) -> Any:
    import torch
    from nvalchemi.data import AtomicData, Batch

    data = AtomicData.from_atoms(atoms, device=device)
    data.add_system_property("energy", torch.zeros((1, 1), dtype=data.positions.dtype, device=device))
    data.add_node_property("forces", torch.zeros_like(data.positions))
    return Batch.from_data_list([data], device=device)


def _largest_force(model: Any, atoms: Atoms, device: torch.device) -> float:
    from .model_support import prepare_model_batch

    batch = prepare_model_batch(_ethanol_batch(atoms, device), model)
    forces = model(batch)["forces"].detach()
    return float(forces.norm(dim=-1).max())


def _relax(model: Any, atoms: Atoms, device: torch.device) -> Atoms:
    """Relax one structure with Toolkit's L-BFGS to ``ETHANOL_FMAX``; return the relaxed structure."""

    from nvalchemi.dynamics import LBFGS, ConvergenceHook, FusedStage

    batch = _ethanol_batch(atoms, device)
    engine = LBFGS(
        model=model,
        n_steps=ETHANOL_MAX_STEPS,
        convergence_hook=ConvergenceHook.from_fmax(ETHANOL_FMAX),
    )
    stage = FusedStage(sub_stages=[(0, engine)], device_type=device.type)
    for hook in model.make_neighbor_hooks():
        stage.register_hook(hook)
    with stage:
        batch = stage.run(batch, n_steps=ETHANOL_MAX_STEPS + 10)
    relaxed = atoms.copy()
    relaxed.positions = batch.positions.detach().cpu().numpy()
    return relaxed


def _config_snapshot(*models: Any) -> list[tuple[Any, set[str]]]:
    seen, snapshot = set(), []
    for model in models:
        for module in model.modules():
            config = getattr(module, "model_config", None)
            if config is not None and id(module) not in seen:
                seen.add(id(module))
                snapshot.append((module, set(config.active_outputs)))
    return snapshot


def single_molecule_check_data(aimnet_model: Any, composed_model: Any) -> dict[str, Any]:
    """Return the largest force on an atom (eV/Å) of ethanol for four models and two relaxed O-H lengths (Å).

    Models: the bare MLIP (``aimnet_model``), the MLIP with D3, the MLIP with
    electrostatics, and both (``composed_model``). The D3 and electrostatics
    variants are built here from ``aimnet_model``, Toolkit's ``DFTD3ModelWrapper``
    and the course's ``DirectCoulombAdapter``. The largest force is the largest
    per-atom vector norm at the reference geometry (ASE ``CH3CH2OH``). The bare
    MLIP and the composed model are then relaxed with L-BFGS to 0.01 eV/Å.
    The learner's models leave with the outputs they came in with.
    """

    from nvalchemi.models import (
        DFTD3ModelWrapper,
        PipelineGroup,
        PipelineModelWrapper,
        PipelineStep,
    )

    from .agent04 import DirectCoulombAdapter, d3_settings

    device = next(aimnet_model.parameters()).device
    atoms = _ethanol()
    snapshot = _config_snapshot(aimnet_model, composed_model)
    try:
        d3 = DFTD3ModelWrapper(**d3_settings()).to(device).eval()
        coulomb = DirectCoulombAdapter().to(device).eval()
        charge_step = PipelineStep(aimnet_model, wire={"charges": "partial_charges"})
        with_d3 = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[aimnet_model], use_autograd=False),
                PipelineGroup(steps=[d3], use_autograd=False),
            ],
            neighbor_adaptation="always",
        ).to(device)
        with_coulomb = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[charge_step, coulomb], use_autograd=True)],
            neighbor_adaptation="always",
        ).to(device)
        aimnet_model.set_config("active_outputs", {"energy", "forces", "charges"})
        d3.set_config("active_outputs", {"energy", "forces"})
        coulomb.set_config("active_outputs", {"energy", "forces"})
        models = {"network": aimnet_model, "d3": with_d3, "coulomb": with_coulomb, "all": composed_model}
        for name, model in models.items():
            model.set_config(
                "active_outputs", {"energy", "forces"} if name == "network" else {"energy", "forces", "charges"}
            )
        force = {name: _largest_force(model, atoms, device) for name, model in models.items()}
        oh = {name: _oh_length(_relax(models[name], atoms, device)) for name in ("network", "all")}
    finally:
        for module, active in snapshot:
            module.model_config.active_outputs = set(active)
    return {"largest_force": force, "oh_relaxed": oh}


def _draw_table(axis: Any, header: Sequence[str], rows: Sequence[Sequence[str]], edges: Sequence[float]) -> None:
    """Draw a table on ``axis``: a bold header in the title size, a rule under each row.

    ``edges`` holds the left edge of each column and the right edge of the last, as shares of the
    width. The first column is aligned left, the others right.
    """

    pal = _PALETTE
    n = len(rows) + 1
    align = ["left"] + ["right"] * (len(header) - 1)

    def x_of(k: int) -> float:
        return edges[k] + 0.02 if align[k] == "left" else edges[k + 1] - 0.02

    for k, text in enumerate(header):
        axis.text(x_of(k), 1 - 0.5 / n, text, ha=align[k], va="center", fontweight="bold",
                  fontsize=plt.rcParams["axes.titlesize"],  # 15 px header size of the style
                  color=pal["ink"], transform=axis.transAxes)
    axis.plot([0, 1], [1 - 1 / n] * 2, color=pal["ink"], lw=1.2, transform=axis.transAxes)
    for r, row in enumerate(rows, 1):
        y = 1 - (r + 0.5) / n
        for k, text in enumerate(row):
            axis.text(x_of(k), y, text, ha=align[k], va="center", color=pal["ink2"], transform=axis.transAxes)
        axis.plot([0, 1], [1 - (r + 1) / n] * 2, color=pal["rule"], lw=0.8, transform=axis.transAxes)


@_styled
def build_single_molecule_table(data: Mapping[str, Any]) -> Any:
    """Draw the ethanol table: bold header in the title size, body in the body size."""

    figure, axis = _figure((len(ETHANOL_LABELS["rows"]) + 1) * _TABLE_ROW_PX / 100 + 0.084)
    axis.axis("off")
    header = ETHANOL_LABELS["header"]
    rows = []
    for key, label in ETHANOL_LABELS["rows"].items():
        oh = data["oh_relaxed"].get(key)
        rows.append(
            [label, f"{data['largest_force'][key]:.2f}", ETHANOL_LABELS["empty"] if oh is None else f"{oh:.3f}"]
        )
    ncol = len(header)
    _draw_table(axis, header, rows, [k / ncol for k in range(ncol + 1)])
    return figure


def show_single_molecule_check(aimnet_model: Any, composed_model: Any) -> None:
    """Show ethanol's largest force on an atom with four models, and O-H after relaxing.

    Takes the learner's bare AIMNet2 model and composed model; builds the D3-only
    and electrostatics-only variants from them with Toolkit pieces.
    """

    data = single_molecule_check_data(aimnet_model, composed_model)
    _show(lambda: build_single_molecule_table(data), ALT_TEXTS["ethanol"])


# ---------------------------------------------------------------------------
# 4. Conformers
# ---------------------------------------------------------------------------
def batch_phi_psi(batch: Any) -> np.ndarray:
    """Return (phi, psi) in degrees for every graph of a batch of alanine dipeptide, shape (n, 2)."""

    from .agent04 import PHI_ATOMS, PSI_ATOMS, dihedral_deg

    positions = _array(batch.positions).reshape(-1, 3)
    n = int(batch.num_graphs)
    per = positions.shape[0] // n
    if per * n != positions.shape[0]:
        raise ValueError("every graph must hold the same molecule")
    frames = positions.reshape(n, per, 3)
    return np.array([(dihedral_deg(p, PHI_ATOMS), dihedral_deg(p, PSI_ATOMS)) for p in frames])


def count_basins(angles: Any, rule: Mapping[str, Any] | Callable[[np.ndarray], int] | None = None) -> int:
    """Count basins among (phi, psi) end points, in degrees, shape (n, 2).

    ``rule`` is a callable that takes the angles and returns the count, or a
    dict with ``tolerance_deg`` (default 25). With the dict, points are taken in
    order; a point joins the first basin whose first point lies within the
    tolerance in both phi and psi (periodic, so -179 and 179 are 2 degrees
    apart), and otherwise starts a new basin.
    """

    rule = DEFAULT_BASIN_RULE if rule is None else rule
    angles = np.asarray(angles, dtype=float).reshape(-1, 2)
    if callable(rule):
        return int(rule(angles))
    tolerance = float(rule["tolerance_deg"])
    centers: list[np.ndarray] = []
    for a in angles:
        if not any(np.abs((a - c + 180.0) % 360.0 - 180.0).max() < tolerance for c in centers):
            centers.append(a)
    return len(centers)


def _graph_atoms(batch: Any, index: int) -> Atoms:
    from ase import Atoms

    numbers = _array(batch.atomic_numbers).reshape(-1).astype(int)
    positions = _array(batch.positions).reshape(-1, 3)
    per = positions.shape[0] // int(batch.num_graphs)
    sl = slice(index * per, (index + 1) * per)
    return Atoms(numbers=numbers[sl], positions=positions[sl], pbc=False)


def conformers_data(
    conformer_batch: Any,
    relaxed_conformers: Any,
    conformer_energy: Any,
    basin_rule: Mapping[str, Any] | Callable[[np.ndarray], int] | None = None,
) -> dict[str, Any]:
    """Return start and end (phi, psi), the basin count and the lowest-energy conformer."""

    start = batch_phi_psi(conformer_batch)
    end = batch_phi_psi(relaxed_conformers)
    energy = _array(conformer_energy).reshape(-1)
    if energy.size != end.shape[0]:
        raise ValueError(f"{energy.size} energies for {end.shape[0]} relaxed conformers")
    lowest = int(np.argmin(energy))
    return {
        "start": start,
        "end": end,
        "n_starts": int(start.shape[0]),
        "k": count_basins(end, basin_rule),
        "lowest": lowest,
        "lowest_atoms": _graph_atoms(relaxed_conformers, lowest),
        "energy": energy,
        "ranking": rank_conformers(end, energy),
    }


def rank_conformers(end: np.ndarray, energy: np.ndarray, count: int = _RANKING_ROWS) -> list[dict[str, Any]]:
    """Return the ``count`` lowest conformers by energy (stable sort), with energy above the lowest in kcal/mol."""

    energy = np.asarray(energy, dtype=float).reshape(-1)
    order = np.argsort(energy, kind="stable")[:count]
    return [
        {
            "rank": rank,
            "index": int(i),
            "phi": float(end[i, 0]),
            "psi": float(end[i, 1]),
            "energy_above_kcal_mol": float((energy[i] - energy.min()) * KCAL_PER_EV),
        }
        for rank, i in enumerate(order, 1)
    ]


def _whole_degrees(value: float) -> str:
    return str(int(np.round(value)) + 0)  # + 0 turns -0 into 0


@_styled
def build_ranking_table(data: Mapping[str, Any]) -> Any:
    """Draw the lowest conformers by energy, drawn like the ethanol table, with a title above."""

    ranking = data["ranking"]
    n = len(ranking) + 1
    figure, axis = _figure((n * _TABLE_ROW_PX + _RANKING_TITLE_PX) / 100 + 0.084)
    axis.axis("off")
    axis.set_title(RANKING_LABELS["title"])
    header = RANKING_LABELS["header"]
    rows = [
        [str(r["rank"]), _whole_degrees(r["phi"]), _whole_degrees(r["psi"]), f"{r['energy_above_kcal_mol']:.2f}"]
        for r in ranking
    ]
    _draw_table(axis, header, rows, np.concatenate([[0.0], np.cumsum(_RANKING_WIDTHS)]))
    return figure


@_styled
def build_conformers_figure(data: Mapping[str, Any]) -> Any:
    """Draw the Ramachandran map of starts, end points and published minima, square axes centred."""

    pal = _PALETTE
    figure, axis = _square_figure(_MAP_BOTTOM_PX, _MAP_LEFT_SHIFT_PX)
    start, end = data["start"], data["end"]
    axis.plot(start[:, 0], start[:, 1], "o", ms=3, mfc="none", mec=pal["ref"], mew=0.8, zorder=2)
    axis.plot(end[:, 0], end[:, 1], "o", ms=5, mfc=pal["nv"], mec=pal["ring"], mew=0.5, zorder=3)
    for key, (phi, psi) in PUBLISHED_MINIMA.items():
        dx, dy = _MINIMA_OFFSETS[key]
        axis.plot(phi, psi, "D", ms=8, mfc="none", mec=pal["ink"], mew=1.4, zorder=4)
        axis.text(phi + dx, psi + dy, CONFORMERS_LABELS["minima"][key], color=pal["ink"], fontweight="bold",
                  zorder=5, bbox={"boxstyle": "round,pad=0.15", "fc": pal["ring"], "ec": "none"})
    axis.set_xlim(-180 - _MAP_PAD_DEG, 180 + _MAP_PAD_DEG)
    axis.set_ylim(-180 - _MAP_PAD_DEG, 180 + _MAP_PAD_DEG)
    axis.set_xticks(range(-180, 181, 90))
    axis.set_yticks(range(-180, 181, 90))
    axis.set_aspect("equal")
    axis.grid(axis="x")
    axis.set_title(CONFORMERS_LABELS["title"].format(n_starts=data["n_starts"], k=data["k"]))
    axis.set_xlabel(CONFORMERS_LABELS["xlabel"])
    axis.set_ylabel(CONFORMERS_LABELS["ylabel"])
    axis.annotate(
        CONFORMERS_LABELS["source"],
        xy=(0, 0),
        xycoords=("axes fraction", axis.xaxis.label),
        xytext=(0, -6),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=plt.rcParams["xtick.labelsize"],  # 11 px footnote size of the style
        color=pal["mute"],
        annotation_clip=False,
    )
    return figure


def show_conformers(
    conformer_batch: Any,
    relaxed_conformers: Any,
    conformer_energy: Any,
    basin_rule: Mapping[str, Any] | Callable[[np.ndarray], int] | None = None,
) -> None:
    """Show the Ramachandran map of the conformer search and the lowest-energy conformer.

    Grey open circles are the starts, green dots the relaxed end points, open
    diamonds published minima. ``conformer_energy`` is (n,) or (n, 1) in eV.
    ``basin_rule`` sets how end points are counted as basins; see ``count_basins``.
    With ``SHOW_RANKING`` a table of the lowest five conformers by energy follows
    the viewer.
    """

    from IPython.display import display

    from .viewers import show_molecule

    data = conformers_data(conformer_batch, relaxed_conformers, conformer_energy, basin_rule)
    _show(lambda: build_conformers_figure(data), ALT_TEXTS["conformers"])
    display(show_molecule(data["lowest_atoms"]))
    if SHOW_RANKING:
        _show(lambda: build_ranking_table(data), ALT_TEXTS["ranking"])
