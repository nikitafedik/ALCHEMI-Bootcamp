# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the Agent 04 step contracts (no GPU, no Toolkit run)."""

from __future__ import annotations

import numpy as np

from helpers import agent04_check as check


class _Coulomb:
    def __init__(self, cutoff):
        self.cutoff = cutoff

    def modules(self):
        return [self]


_Coulomb.__name__ = "DirectCoulombAdapter"


class _D3:
    def modules(self):
        return [self]


_D3.__name__ = "DFTD3ModelWrapper"


class _Pipeline:
    def __init__(self, *parts):
        self.parts = parts

    def modules(self):
        return [self, *self.parts]


def test_steps_run_in_contract_order():
    assert check._order("conformers") == ["compose_build", "compose_run", "conformers"]
    assert check._order("no_d3") == ["compose_build", "compose_run", "no_d3"]


def test_every_step_names_its_file_and_checks():
    for name, step in check.STEPS.items():
        assert step.file.endswith(".py"), name
        assert step.names and step.checks, name


def test_energy_accepts_both_shapes_and_rejects_others():
    assert check._energy_problems({"e": np.zeros((90, 1))}, "e", 90) == []
    assert check._energy_problems({"e": np.zeros(90)}, "e", 90) == []
    assert "shape (45, 2)" in check._energy_problems({"e": np.zeros((45, 2))}, "e", 90)[0]
    assert "not finite" in check._energy_problems({"e": np.full((90, 1), np.nan)}, "e", 90)[0]


def test_coulomb_range_below_default_is_reported():
    short = {"m": _Pipeline(_Coulomb(8.0))}
    assert "8 Å" in check._coulomb_problems(short, "m")[0]
    assert check._coulomb_problems({"m": _Pipeline(_Coulomb(15.0))}, "m") == []


def test_no_d3_model_must_have_electrostatics_and_no_d3():
    with_d3 = {"no_d3_model": _Pipeline(_Coulomb(15.0), _D3()), "no_d3_energy": np.zeros((90, 1))}
    assert any("still has a D3 term" in p for p in check._check_no_d3(with_d3))
    without = {"no_d3_model": _Pipeline(_Coulomb(15.0)), "no_d3_energy": np.zeros((90, 1))}
    assert check._check_no_d3(without) == []
    no_coulomb = {"no_d3_model": _Pipeline(), "no_d3_energy": np.zeros((90, 1))}
    assert any("no electrostatics" in p for p in check._check_no_d3(no_coulomb))


def test_interaction_energy_uses_ab_a_b_order():
    energy = np.array([[-10.0], [-4.0], [-5.0]])  # AB, A, B in eV
    assert np.isclose(check._interaction_kcal(energy)[0], (-10.0 + 4.0 + 5.0) * check.KCAL_PER_EV)


def test_main_rejects_an_unknown_step(capsys):
    assert check.main(["check", "nope"]) == 2
    assert "usage" in capsys.readouterr().out


def test_fine_control_range_of_12_passes_and_failed_the_old_floor(monkeypatch):
    model = {"m": _Pipeline(_Coulomb(12.0))}
    assert check._coulomb_problems(model, "m") == []
    monkeypatch.setattr(check, "MIN_COULOMB_RANGE", 15.0)  # the floor before 26 Nov
    assert "12 Å; use at least 15 Å" in check._coulomb_problems(model, "m")[0]


class _Step:
    def __init__(self, model):
        self.model = model


class _Group:
    def __init__(self, use_autograd, *models):
        self.use_autograd = use_autograd
        self.steps = [_Step(m) for m in models]


class _Grouped:
    def __init__(self, *groups):
        self.groups = list(groups)


def test_coulomb_term_must_sit_in_an_autograd_group():
    good = {"m": _Grouped(_Group(True, object(), _Coulomb(15.0)), _Group(False, _D3()))}
    assert check._autograd_problems(good, "m") == []
    bad = {"m": _Grouped(_Group(False, object(), _Coulomb(15.0)), _Group(False, _D3()))}
    assert "outside the summed energy" in check._autograd_problems(bad, "m")[0]
    assert check._autograd_problems({"m": object()}, "m") == []  # not a pipeline: nothing to say


def test_threshold_is_part_of_the_conformer_contract():
    assert check.THRESHOLD_IN_CONTRACT is True
    assert "{threshold" in check.LINES["not_relaxed"] and "relax_settings" in check.LINES["not_relaxed"]


class _AIMNet2Wrapper:
    """Stands for the bare wrapper, whose network holds a Coulomb module of its own."""

    def modules(self):
        return [self, _Coulomb(4.6)]


def test_terms_are_pipeline_steps_not_inner_modules():
    bare = _AIMNet2Wrapper()
    assert check._class_names(bare) == ["_AIMNet2Wrapper"]
    assert not check._has_electrostatics(bare)
    assert any("no electrostatics" in p for p in check._check_no_d3({"no_d3_model": bare, "no_d3_energy": np.zeros((90, 1))}))
    composed = _Grouped(_Group(True, bare, _Coulomb(15.0)), _Group(False, _D3()))
    assert check._class_names(composed) == ["_AIMNet2Wrapper", "DirectCoulombAdapter", "DFTD3ModelWrapper"]
    assert check._has_electrostatics(composed)
    assert check._coulomb_problems({"m": composed}, "m") == []  # the inner 4.6 Å module is not a term


def test_toolkit_electrostatics_classes_count_as_electrostatics():
    class EwaldModelWrapper:
        pass

    assert check._has_electrostatics(_Grouped(_Group(True, _AIMNet2Wrapper(), EwaldModelWrapper())))

