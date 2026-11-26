# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from ase import Atoms
from ase.build import molecule

_NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NOTEBOOK_DIR))

from helpers import relaxation_lbfgs


@dataclass(frozen=True)
class _Spec:
    structure: Atoms
    display_name: str

    def build(self) -> Atoms:
        return self.structure.copy()


class _Selection:
    def selected_specs(self):
        return [_Spec(molecule("H2O"), "Water")]


class _StopAfterLoad(Exception):
    pass


def _lbfgs_available() -> bool:
    try:
        from nvalchemi.dynamics import LBFGS  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _lbfgs_available(), reason="LBFGS needs Toolkit 0.3.0")
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, False),
        ({"speed": False}, False),
        ({"speed": True}, True),
        ({"speed": True, "compile_model": False}, False),
        ({"compile_model": True}, True),
    ],
)
def test_speed_switch_sets_compile_model_on_the_checkpoint_load(
    monkeypatch, kwargs, expected
) -> None:
    models = pytest.importorskip("nvalchemi.models")
    seen = {}

    def fake_from_checkpoint(*args, **load_kwargs):
        seen.update(load_kwargs)
        raise _StopAfterLoad

    monkeypatch.setattr(models.MACEWrapper, "from_checkpoint", fake_from_checkpoint)
    monkeypatch.setattr(relaxation_lbfgs, "mace_checkpoint_path", lambda name: name)
    with pytest.raises(_StopAfterLoad):
        relaxation_lbfgs.relax_selected_structures_lbfgs(
            _Selection(), device="cpu", batch_multiplier=1, **kwargs
        )
    assert seen["compile_model"] is expected
    assert seen["enable_cueq"] is False  # CPU device: never cuEq unless asked


@pytest.mark.parametrize(
    ("kwargs", "device", "installed", "expected"),
    [
        ({"speed": False}, "cuda", True, (False, False)),
        ({"speed": True}, "cuda", True, (True, True)),
        ({"speed": True}, "cuda", False, (True, False)),
        ({"speed": True}, "cpu", True, (True, False)),
        ({"speed": True, "enable_cueq": False}, "cuda", True, (True, False)),
        ({"speed": True, "compile_model": False}, "cuda", True, (False, True)),
        ({"speed": False, "enable_cueq": True}, "cuda", False, (False, True)),
    ],
)
def test_speed_settings_resolve_cueq_only_where_it_can_run(
    monkeypatch, kwargs, device, installed, expected
) -> None:
    monkeypatch.setattr(relaxation_lbfgs, "cueq_kernels_installed", lambda: installed)
    settings = {"compile_model": None, "enable_cueq": None, **kwargs}
    assert relaxation_lbfgs.resolve_speed_settings(device=device, **settings) == expected


@pytest.mark.skipif(not _lbfgs_available(), reason="LBFGS needs Toolkit 0.3.0")
@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="needs a CUDA device"
)
def test_cell9_default_call_stays_plain_even_with_cueq_kernels(monkeypatch) -> None:
    """Cell 9's call (module00, no speed keywords) loads MACE without cuEq or compile,
    also in a runtime where the cuEquivariance kernels are installed."""

    from helpers import module00

    models = pytest.importorskip("nvalchemi.models")
    seen = {}

    def fake_from_checkpoint(*args, **load_kwargs):
        seen.update(load_kwargs)
        raise _StopAfterLoad

    monkeypatch.setattr(models.MACEWrapper, "from_checkpoint", fake_from_checkpoint)
    monkeypatch.setattr(relaxation_lbfgs, "mace_checkpoint_path", lambda name: name)
    monkeypatch.setattr(relaxation_lbfgs, "cueq_kernels_installed", lambda: True)
    with pytest.raises(_StopAfterLoad):
        module00.relax_selected_structures(
            _Selection(),
            device="cuda",
            fmax=0.05,
            max_steps=500,
            distortion_std_angstrom=0.20,
            distortion_seed=7,
            batch_multiplier=12,
        )
    assert seen["enable_cueq"] is False
    assert seen["compile_model"] is False
