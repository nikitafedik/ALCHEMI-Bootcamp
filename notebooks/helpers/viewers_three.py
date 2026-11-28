# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""three.js versions of the Core structure explorer and relaxation player.

The browser code lives in ``shared/alchemi-viewer`` and the vendored three.js
build in ``shared/three-0.186.1``. Both are read once, inlined into an iframe
``srcdoc``, and run without network access. Selection stays in Python.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from html import escape
from typing import Any
from uuid import uuid4

import numpy as np
from ase import Atoms

from . import viewers
from .viewers import (
    MAX_RELAXATION_SYSTEMS,
    _json_for_script,
    _metadata_from_spec,
)

_THREE_BUNDLE = viewers._PROJECT_ROOT / "shared" / "three-0.186.1" / "three.min.js"
_VIEWER_JS = viewers._PROJECT_ROOT / "shared" / "alchemi-viewer" / "alchemi-viewer.js"
_VIEWER_CSS = viewers._PROJECT_ROOT / "shared" / "alchemi-viewer" / "alchemi-viewer.css"

_GREEN = viewers._GREEN
_CARD_HEIGHT = 164
_PLAYER_HEIGHT = 200
_PLAYER_FRAME_HEIGHT = 330


@lru_cache(maxsize=1)
def _viewer_assets() -> tuple[str, str, str]:
    """Read the three.js build, the viewer script and its CSS once."""

    return (
        _THREE_BUNDLE.read_text(encoding="utf-8"),
        _VIEWER_JS.read_text(encoding="utf-8"),
        _VIEWER_CSS.read_text(encoding="utf-8"),
    )


def _script_text(source: str) -> str:
    """Keep a vendored script from closing its own script tag."""

    return source.replace("</script", "<\\/script")


def _viewer_document(body: str, mount_script: str) -> str:
    """Return one self-contained document that loads the viewer and mounts it."""

    three, viewer, css = _viewer_assets()
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<style>html,body{margin:0;background:transparent}"
        "body{overflow:hidden;font-family:'NVIDIA Sans',Arial,sans-serif}"
        "body{color:#1A1A1A}</style>"
        f"<style>{css}</style>"
        f"<script>{_script_text(three)}</script>"
        f"<script>{_script_text(viewer)}</script>"
        f"</head><body>{body}<script>{mount_script}</script></body></html>"
    )


# The course is light only. A srcdoc iframe stays transparent only when its
# color-scheme matches the page around it, so the root element is set to light.
_THEME_TO_BODY = "document.documentElement.style.colorScheme='light';"

# Grow or shrink the notebook iframe to the content height (same-origin srcdoc).
_AUTO_HEIGHT = (
    "(function(){var f=window.frameElement;if(!f||!window.ResizeObserver)return;"
    "var root=document.getElementById('root');"
    "var fit=function(){f.style.height=Math.ceil(root.getBoundingClientRect().height)+'px';};"
    "new ResizeObserver(fit).observe(root);fit();})();"
)


def _theme_option() -> str:
    """Return the viewer's ``theme`` option; the course is light only."""

    return ",theme:\"light\""


def _card_iframe(structure: Mapping[str, Any], *, height: int) -> str:
    """Return the notebook iframe that shows one rotatable structure."""

    mount = (
        "AlchemiViewer.mountStructure(document.getElementById('root'),"
        f"{_json_for_script(structure)},{{height:{height}{_theme_option()}}});" + _THEME_TO_BODY
    )
    document = _viewer_document("<div id='root'></div>", mount)
    return (
        f'<iframe title="Interactive 3D structure: '
        f'{escape(str(structure["name"]), quote=True)}" '
        f'srcdoc="{escape(document, quote=True)}" style="display:block;width:100%;'
        f'height:{height}px;border:0;background:transparent;" loading="eager"></iframe>'
    )


def _rounded(values: Any, decimals: int = 4) -> list[Any]:
    return np.round(np.asarray(values, dtype=float), decimals).tolist()


def _cell_of(atoms: Atoms) -> list[list[float]] | None:
    if not bool(np.asarray(atoms.pbc).any()):
        return None
    return _rounded(atoms.cell.array)


def _edges_of(atoms: Atoms) -> list[dict[str, list[float]]]:
    return [
        {"start": _rounded(edge["start"]), "end": _rounded(edge["end"])}
        for edge in viewers._cell_edges(atoms)
    ]


def _structure_data(item: Mapping[str, Any], atoms: Atoms) -> dict[str, Any]:
    """Build the JSON the viewer expects for one explorer card."""

    return {
        "name": item["display_name"],
        "formula": item["formula"],
        "atom_count": item["atom_count"],
        "symbols": list(item["symbols"]),
        "positions": _rounded(atoms.positions),
        "periodic": bool(item["periodic"]),
        "cell": _cell_of(atoms),
        "cell_edges": _edges_of(atoms),
    }


def _picker_css() -> str:
    """Artist CSS for the picker; the picker uses the light variables (``.av-light``)."""

    css = _viewer_assets()[2]
    return (
        f"<style>{css}\n"
        ".av-picker .widget-html:not(.av-chip),.av-picker .widget-html-content,"
        ".av-picker .widget-vbox,.av-picker .widget-hbox,.av-picker .widget-gridbox{"
        "color:inherit}\n"
        ".av-picker .widget-html,.av-picker .widget-html-content{"
        "overflow:visible;box-sizing:border-box}\n"
        ".av-picker .av-chip,.av-picker .av-chip .widget-html-content,"
        ".av-picker .av-btn.is-on{color:#FFFFFF}\n"
        ".av-picker .av-chip.is-empty,.av-picker .av-chip.is-empty .widget-html-content,"
        ".av-picker .av-btn:disabled{color:var(--av-mute)}\n"
        ".av-picker .av-btn:not(.is-on):not(:disabled){color:var(--av-ink)}\n"
        ".av-picker .av-card{box-sizing:border-box;overflow:hidden}\n"
        ".av-picker iframe{display:block;max-width:100%}\n"
        "@media(max-width:760px){.av-picker .av-grid{"
        "grid-template-columns:repeat(2,minmax(0,1fr)) !important}}\n"
        "@media(max-width:520px){.av-picker .av-grid{"
        "grid-template-columns:1fr !important}}</style>"
    )


class StructureSelectionThree(viewers.StructureSelection):
    """Structure picker whose cards are three.js viewers.

    Python behaviour matches ``viewers.StructureSelection``: toggle buttons,
    at most five selected systems, ``selected_keys`` and ``selected_specs``.
    Controls use the viewer's own classes (``av-btn``, ``av-card``, ``av-chip``).
    """

    def __init__(
        self,
        catalog: Mapping[str, Any],
        items: Sequence[dict[str, Any]],
        selected_keys: Sequence[str],
        structures: Mapping[str, Atoms],
    ) -> None:
        import ipywidgets as widgets

        self._catalog = dict(catalog)
        self._items = list(items)
        self._checkboxes: dict[str, Any] = {}
        self._cards: dict[str, Any] = {}
        self._status = widgets.HTML(layout=widgets.Layout(width="auto"))
        self._status.add_class("av-chip")

        for item in items:
            key = item["key"]
            control = widgets.ToggleButton(
                value=key in selected_keys,
                description=(
                    "✓ In batch" if key in selected_keys else "+ Add to batch"
                ),
                layout=widgets.Layout(width="100%"),
            )
            control.add_class("av-btn")
            control.observe(
                lambda change, item_key=key: self._toggle(item_key, change),
                "value",
            )
            self._checkboxes[key] = control
            viewer = widgets.HTML(
                value=_card_iframe(
                    _structure_data(item, structures[key]), height=_CARD_HEIGHT
                ),
                layout=widgets.Layout(width="100%"),
            )
            facts = widgets.HTML(
                value=(
                    '<div class="av-head"><strong>'
                    f"{escape(item['display_name'])}</strong>"
                    f'<span class="av-muted">{item["atom_count"]} atoms</span></div>'
                    f'<div class="av-muted">{escape(item["purpose"])} · '
                    f"{escape(item['domain'])}</div>"
                ),
                layout=widgets.Layout(width="100%", min_height="64px", flex="1 1 auto"),
            )
            card = widgets.VBox(
                [viewer, facts, control],
                layout=widgets.Layout(width="100%", height="100%"),
            )
            card.add_class("av-card")
            self._cards[key] = card

        title = widgets.HTML(
            value=(
                "<div><strong>Choose systems for the batch</strong>"
                '<span class="av-muted" style="display:block;margin-top:3px">'
                "Drag to rotate, scroll "
                f"to zoom, choose up to {MAX_RELAXATION_SYSTEMS} systems</span></div>"
            )
        )
        header = widgets.HBox(
            [title, self._status],
            layout=widgets.Layout(width="100%", align_items="center"),
        )
        header.add_class("av-pick-head")
        grid = widgets.GridBox(
            list(self._cards.values()),
            layout=widgets.Layout(
                width="100%",
                grid_template_columns="repeat(4, minmax(0, 1fr))",
                grid_gap="10px",
            ),
        )
        grid.add_class("av-grid")

        styles = widgets.HTML(_picker_css())
        self.widget = widgets.VBox(
            [styles, header, grid], layout=widgets.Layout(width="100%")
        )
        self.widget.add_class("alchemi-structure-picker")
        self.widget.add_class("av-picker")
        self.widget.add_class("av-light")
        self._static_fallback = self._build_static_fallback()
        self._sync()

    def _sync(self) -> None:
        """Mirror the selection onto control, card, chip and disabled states."""

        chosen = set(self.selected_keys)
        full = len(chosen) >= MAX_RELAXATION_SYSTEMS
        for key, control in self._checkboxes.items():
            on = key in chosen
            control.description = "✓ In batch" if on else "+ Add to batch"
            control.disabled = full and not on
            for widget, name in ((control, "is-on"), (self._cards[key], "is-selected")):
                if on:
                    widget.add_class(name)
                else:
                    widget.remove_class(name)
        self._update_status()

    def _toggle(self, key: str, change: Mapping[str, Any]) -> None:
        if bool(change["new"]) and len(self.selected_keys) > MAX_RELAXATION_SYSTEMS:
            self._checkboxes[key].value = False
            return
        self._sync()

    def _update_status(self) -> None:
        count = len(self.selected_keys)
        label = "system" if count == 1 else "systems"
        self._status.value = (
            f"{count} {label} in batch" if count else "Choose one or more systems"
        )
        if count:
            self._status.remove_class("is-empty")
        else:
            self._status.add_class("is-empty")


def show_structure_explorer_three(
    catalog: Mapping[str, Any],
    *,
    selected: str | Sequence[str] | None = None,
) -> StructureSelectionThree:
    """Show a selectable three.js molecular grid linked to later Python cells.

    ``catalog`` accepts the ordered mapping returned by ``opening_performance_catalog``.
    """

    if not catalog:
        raise ValueError("catalog must contain at least one structure")
    items = []
    structures: dict[str, Atoms] = {}
    for key, spec in catalog.items():
        if isinstance(spec, Atoms) or not callable(getattr(spec, "build", None)):
            raise TypeError(f"catalog entry {key!r} must provide a build() method")
        atoms = spec.build()
        if not isinstance(atoms, Atoms):
            raise TypeError(f"catalog entry {key!r} did not build an ASE Atoms object")
        items.append(_metadata_from_spec(str(key), spec, atoms))
        structures[str(key)] = atoms

    valid_keys = {item["key"] for item in items}
    if selected is None:
        selected_keys = []
    elif isinstance(selected, str):
        selected_keys = [selected]
    else:
        selected_keys = list(selected)
    missing = [key for key in selected_keys if key not in valid_keys]
    if missing:
        raise KeyError(f"selected structures are absent from catalog: {missing}")
    if len(set(selected_keys)) > MAX_RELAXATION_SYSTEMS:
        raise ValueError(
            f"select at most {MAX_RELAXATION_SYSTEMS} systems for the synchronized player"
        )
    return StructureSelectionThree(catalog, items, selected_keys, structures)


def _player_iframe_three(
    items: Sequence[Mapping[str, Any]],
    *,
    frame_interval_ms: int,
    step_labels: Sequence[int] | None = None,
) -> str:
    """Return one iframe that mounts the synchronized three.js player."""

    labels_option = (
        f",step_labels:{_json_for_script(list(step_labels))}" if step_labels else ""
    )
    mount = (
        "AlchemiViewer.mountPlayer(document.getElementById('root'),"
        f"{_json_for_script(list(items))},{{height:{_PLAYER_HEIGHT},"
        f"frame_interval_ms:{int(frame_interval_ms)}{labels_option}{_theme_option()}}});"
        + _THEME_TO_BODY
        + _AUTO_HEIGHT
    )
    document = _viewer_document("<div id='root'></div>", mount)
    return (
        '<iframe title="Synchronized relaxation player" '
        f'srcdoc="{escape(document, quote=True)}" style="display:block;width:100%;'
        f'height:{_PLAYER_FRAME_HEIGHT}px;border:0;background:transparent;" '
        'loading="eager"></iframe>'
    )


def show_relaxation_player_three(
    trajectories: Mapping[str, Sequence[Any]] | Sequence[Any],
    *,
    labels: Sequence[str] | None = None,
    force_history: Mapping[str, Sequence[float]] | None = None,
    device: str = "device not recorded",
    elapsed_s: float | None = None,
    convergence: Mapping[str, str] | str | None = None,
    batch_multiplier: int = 1,
    frame_interval_ms: int = 120,
    reference_structures: Mapping[str, Atoms] | Sequence[Atoms] | None = None,
    step_labels: Sequence[int] | None = None,
    converged_steps: Mapping[str, int | None] | None = None,
) -> Any:
    """Show synchronized relaxation frames with force-coloured atoms (three.js).

    ``step_labels`` gives the real step number of each frame when frames were
    thinned or recorded every few steps; the step label then shows those numbers.
    ``converged_steps`` maps a system name to the step at which it reached the
    force threshold (``None`` or 0 if it never did); the player marks that frame.

    Takes the arguments of ``viewers.show_relaxation_player``. When the frames
    carry no forces and no ``force_history`` is given, the three.js player has
    nothing to colour or read out, so the 3Dmol player is returned instead.
    """

    atom_force_history = viewers._normalise_atom_force_magnitudes(trajectories, labels)
    normalised = viewers._normalise_trajectories(trajectories, labels)
    if len(normalised) > MAX_RELAXATION_SYSTEMS:
        raise ValueError(
            "the Core relaxation player supports at most "
            f"{MAX_RELAXATION_SYSTEMS} systems"
        )
    if (
        isinstance(batch_multiplier, bool)
        or not isinstance(batch_multiplier, int)
        or batch_multiplier < 1
    ):
        raise ValueError("batch_multiplier must be a positive integer")
    if (
        isinstance(frame_interval_ms, bool)
        or not isinstance(frame_interval_ms, int)
        or frame_interval_ms < 16
    ):
        raise ValueError("frame_interval_ms must be an integer of at least 16")
    if atom_force_history is None and force_history is None:
        return viewers.show_relaxation_player(
            trajectories,
            labels=labels,
            force_history=force_history,
            device=device,
            elapsed_s=elapsed_s,
            convergence=convergence,
            batch_multiplier=batch_multiplier,
            frame_interval_ms=frame_interval_ms,
            reference_structures=reference_structures,
        )

    names = list(normalised)
    total_graphs = len(names) * batch_multiplier
    atom_counts = {name: len(frames[0]) for name, frames in normalised.items()}
    total_atoms = sum(atom_counts.values()) * batch_multiplier
    # Educator's wording (2026-11-25): "Batch: 5 systems × 12 copies = 60 structures, 2,976 atoms".
    system_label = "system" if len(names) == 1 else "systems"
    copy_label = "copy" if batch_multiplier == 1 else "copies"
    structure_label = "structure" if total_graphs == 1 else "structures"
    atom_label = "atom" if total_atoms == 1 else "atoms"
    batch_status = (
        f"Batch: {len(names)} {system_label} × {batch_multiplier} "
        f"{copy_label} = {total_graphs:,} {structure_label}, "
        f"{total_atoms:,} {atom_label}"
    )
    references = viewers._reference_structure_map(reference_structures, names)
    frame_count = len(next(iter(normalised.values())))
    forces = viewers._force_values(force_history, names, frame_count)

    items = []
    for name, frames in normalised.items():
        reference = references.get(name, frames[0])
        if atom_force_history is not None:
            per_atom = atom_force_history[name]
            fmax = [round(float(max(values)), 4) for values in per_atom]
        else:
            per_atom = [[0.0] * len(frames[0]) for _ in frames]
            fmax = [round(float(value), 4) for value in forces[name]]
        stop_frame = None
        step_of = (converged_steps or {}).get(name)
        if step_of:
            numbers = list(step_labels) if step_labels is not None else list(range(frame_count))
            stop_frame = next((k for k, n in enumerate(numbers) if n >= step_of), None)
        items.append(
            {
                "name": name,
                "atom_count": atom_counts[name],
                "symbols": frames[0].get_chemical_symbols(),
                "cell": _cell_of(reference),
                "cell_edges": _edges_of(reference),
                "frames": [_rounded(frame.positions) for frame in frames],
                "atom_forces": [_rounded(values) for values in per_atom],
                "fmax": fmax,
                **({"stop_frame": stop_frame} if stop_frame is not None else {}),
            }
        )
    del device, elapsed_s, convergence

    root_id = f"alchemi-relax-{uuid4().hex}"
    static_card_parts = []
    for name, frames in normalised.items():
        values = forces[name]
        force_text = (
            "force history not recorded"
            if values[0] is None
            else f"fmax {values[0]:.3f} → {values[-1]:.3f} eV/Å"
        )
        static_card_parts.append(
            f'<section style="min-width:0;padding:0 10px 12px;overflow:hidden;'
            f'border-right:1px solid {viewers._BORDER};'
            f'border-bottom:1px solid {viewers._BORDER}">'
            + viewers._static_pair_svg(
                frames[0], frames[-1], title=f"{name}: initial and final"
            )
            + '<div style="padding:8px 2px 0;font-size:.78rem">'
            + f"{escape(name)} · {atom_counts[name]} atoms · "
            + f"{escape(force_text)}</div></section>"
        )
    static_cards = "".join(static_card_parts)
    if step_labels is not None and len(step_labels) != frame_count:
        raise ValueError("step_labels must hold one step number per frame")
    player_iframe = _player_iframe_three(
        items, frame_interval_ms=frame_interval_ms, step_labels=step_labels
    )
    markup = f"""
<div id="{root_id}" aria-label="Synchronized relaxation player" data-batch-multiplier="{batch_multiplier}" data-total-graphs="{total_graphs}" data-total-atoms="{total_atoms}" style="max-width:none">
  {viewers._common_css(root_id)}
  <style>#{root_id}{{background:transparent;color:inherit;border:0}}</style>
  <div data-static-fallback style="padding:12px">
    {viewers.runtime_required_notice()}
    <p style="margin:0 2px 10px;font-size:.82rem;font-weight:650">{batch_status}</p>
    <div class="alchemi-grid" style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0">{static_cards}</div>
    <p class="alchemi-muted" style="margin:10px 2px 0;font-size:.82rem">Static initial/final preview. Run this cell in Jupyter to play synchronized frames.</p>
  </div>
  <div data-interactive>
    <p style="margin:0 2px 8px;font-size:.82rem;font-weight:650">{batch_status}</p>
    {player_iframe}
  </div>
</div>
<script>(function(){{
const root=document.getElementById('{root_id}');
if(root&&root.querySelector('iframe'))root.classList.add('is-live');
}})();</script>
"""
    return viewers._html(markup)
