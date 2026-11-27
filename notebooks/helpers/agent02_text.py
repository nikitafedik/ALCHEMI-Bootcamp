# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every line a learner reads from the helpers and the step checks of notebook 02, in one place.

The helper (``agent02.py``) and the step checks (``agent02_checks.py``) take their words from here,
so a change of wording is an edit in this file only.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# The choices of "Try it: CPU first, or straight to the GPU" (the build_routes step). This block is
# the one place to change them: the helper, the step check and the tests read these names.
# The sample file ``solutions/agent-02/build_routes.py`` repeats the size, the warm-up and the
# repeats as plain numbers (a learner's file cannot import this module); a test keeps them equal.
# ---------------------------------------------------------------------------
# True: build_routes takes the place of "Try it: a batch of your choice" (the own_batch step), and the
# numbered steps 1 to 6 end with zarr_roundtrip. False: own_batch is numbered too, as step 7.
# The check of own_batch runs by name either way.
PROVISIONAL_BUILD_ROUTES_REPLACES_OWN_BATCH = True
# The timing batch: 1,000 structures, one warm-up build, the best of five timed builds.
PROVISIONAL_N_STRUCTURES = 1000
PROVISIONAL_WARMUPS = 1
PROVISIONAL_TIMED_RUNS = 5
# What the result table says about the route from tensors.
PROVISIONAL_TENSOR_NOTE = "quicker still, but nothing is filled in or checked for you"
# The route from tensors carries these two fields only, and the table says so.
PROVISIONAL_TENSOR_FIELDS = ("positions", "atomic_numbers")
PROVISIONAL_TENSOR_FIELDS_WORDS = "positions and atomic numbers only"
# True: an order other than tensors < CPU then moved < directly on the GPU is marked "unexpected"
# and the step passes. False: that order fails the step.
PROVISIONAL_UNEXPECTED_ORDER_PASSES = True

# ---------------------------------------------------------------------------
# show_session()
# ---------------------------------------------------------------------------
SESSION = (
    "Give these to your agent when you start the session:\n"
    "  notebook    notebooks/alchemi-agent-02-data-and-batching.ipynb\n"
    "  helpers     from helpers import agent02 as helpers\n"
    "  work in     notebooks/agent02_work/\n"
    "  check with  python -m helpers.agent02 check <step>\n"
    "  each file   runs in a notebook cell (%load), after the earlier files\n"
    "  always      run the check from notebooks/ after you write a file, and fix\n              what it reports; <step> is the file name without .py"
)

# ---------------------------------------------------------------------------
# show_fields(data): the field table
# ---------------------------------------------------------------------------
FIELD_HEADERS = ["Field", "Shape", "Unit", "Belongs to", "First values", "Device"]
LEVEL_NAMES = {"node": "atoms", "edge": "pairs", "system": "structure"}
NOT_SET = "(not set yet)"  # a field the model call fills in later
NO_DEVICE = "-"
UNITS = {
    "positions": "Å",
    "forces": "eV/Å",
    "energy": "eV",
    "cell": "Å",
    "stress": "eV/Å³",
    "atomic_masses": "amu",
    "velocities": "√(eV/amu)",
    "charges": "e",
    "charge": "e",
    "dipole": "e·Å",
}
UNIT_DIMENSIONLESS = "1"  # counts and labels: atomic_numbers, atom_categories

# ---------------------------------------------------------------------------
# show_results(batch, energy, forces)
# ---------------------------------------------------------------------------
RESULTS_HEADERS = ["Index", "Molecule", "Atoms", "Energy [eV]", "Largest force [eV/Å]"]

# ---------------------------------------------------------------------------
# show_store(...)
# ---------------------------------------------------------------------------
STORE_COUNT = "{structures} structures, {atoms} atoms"
STORE_HEADERS = ["Field", "Belongs to", "Shape", "Unit", "Type"]
STORE_PER_STRUCTURE_HEADERS = ["Index", "Atoms", "Energy [eV]", "Largest force [eV/Å]"]
STORE_EQUAL = "read back equals saved"
STORE_DIFFERS = "read back differs from saved"

# ---------------------------------------------------------------------------
# show_build_routes(build_table): the build-routes result and its explanation
# ---------------------------------------------------------------------------
# The three rows of the table: route -> (its name, what it tells you).
BUILD_ROUTE_TEXT = {
    "tensors": ("from tensors", PROVISIONAL_TENSOR_NOTE),
    "cpu_then_moved": ("from objects on the CPU, then moved", "the fastest route from objects: one move"),
    "direct_gpu": ("from objects directly on the GPU", "slower: one small move per structure"),
}
BUILD_TENSOR_FIELDS_MARK = " ({words})"  # added to the tensor route's name: "from tensors (positions and atomic numbers only)"
BUILD_HEADERS = ["Route", "Time", "What it tells you"]
BUILD_UNEXPECTED_MARK = "  (unexpected)"
BUILD_ORDER_LINE = "Order check: tensors < CPU then moved < directly on the GPU"
BUILD_ORDER_USUAL = "as usual"
BUILD_ORDER_UNEXPECTED = "unexpected"
BUILD_SIZE_LINE = "One batch of {structures:,} structures, {atoms:,} atoms"
BUILD_REPEATS_PART = "; {runs} timed builds per route"  # added when the table says how many builds were timed (best, mean or median: the table's choice)
BUILD_API_BOX = (
    "> </> API\n"
    ">\n"
    "> `Batch.from_raw_dicts([...], device=...)`\n"
    ">\n"
    "> **Input:** raw tensors you trust, structure by structure\n"
    "> **Effect:** builds one `Batch`, without `AtomicData` and without validation"
)

# ---------------------------------------------------------------------------
# The step checks: `python -m helpers.agent02 check <step>`. One plain line per problem.
# ---------------------------------------------------------------------------
CHECK_LINES = {
    # any step
    "ok": "{step}: OK ({n} names)",
    "unknown_step": "unknown step {name}; steps: {steps}",
    "missing_name": "{step}: {name} is missing; the step asks for it",
    "earlier_failed": "{step}: an earlier file failed: {file}: {error}",
    "run_failed": "{step}: running {file} failed: {error}",
    "run_failed_second": "{step}: running {file} a second time failed: {error}",
    "store_exists": "{step}: a store already exists at the path the file writes to; use a fresh path or remove the old store first",
    "name_error": "{step}: {file} uses {name}, which is not defined in an earlier file; define it in this file",
    "cell_only_name": "{step}: {file} uses {name}, which does not exist in a notebook cell; remove that line",  # __file__
    "file_missing": "agent02_work/{file} not found",
    "changed_name": "{step}: {file} changed {name}, which the later steps use; give what you build here another name",
    # atomic_data
    "caffeine_type": "{step}: caffeine_data is {type}; it should be an AtomicData",
    "caffeine_atoms": "{step}: caffeine_data has {n} atoms; caffeine has {expected}",
    # batch
    "molecule_data_type": "{step}: molecule_data should be a list of AtomicData",
    "molecule_data_counts": "{step}: molecule_data has atom counts {got}; caffeine, aspirin, ibuprofen, paracetamol and capsaicin give {expected}, in any order",
    "batch_type": "{step}: batch is {type}; it should be a Batch",
    "batch_counts": "{step}: batch has {g} structures with atom counts {got}; the five molecules give {expected}, in any order",
    # my_batch
    "my_batch_type": "{step}: my_batch is {type}; it should be a Batch",
    "my_batch_size": "{step}: my_batch has {g} structures; the five molecules plus yours give {expected}",
    "my_batch_order": "{step}: my_batch has atom counts {got}; removing your molecule should leave the five of molecule_data in their order",
    # model_call and own_batch
    "own_batch_type": "{step}: own_batch is {type}; it should be a Batch",
    "not_tensor": "{step}: {name} is {type}; it should be a tensor",
    "tensor_shape": "{step}: {name} has shape {got}; {why} gives {expected}",
    "not_finite": "{step}: {name} has values that are not finite",
    "forces_sum": "{step}: the forces of one molecule sum to {worst:.3g} eV/Å instead of 0; forces and atoms may be out of order",
    # zarr_roundtrip and own_batch
    "store_path_missing": "{step}: {name} {path} does not exist",
    "store_too_small": "{step}: the store holds {n} structures; {name} has {m}",
    "loaded_type": "{step}: loaded_batch is {type}; it should be a Batch",
    "loaded_size": "{step}: loaded_batch has {g} structures and {a} atoms; batch has {g2} and {a2} or more",
    "loaded_no_field": "{step}: loaded_batch has no {name}; save it with the batch",
    "saved_no_field": "{step}: batch has no {name} to compare with; attach it before saving",
    "loaded_differs": "{step}: loaded_batch {name} differs from batch",
    "store_count_own": "{step}: the store holds {n} structures; own_batch has {m}",
    # build_routes
    "table_type": "{step}: build_table is {type}; it should be a table (a pandas DataFrame or a list of rows)",
    "table_rows": "{step}: build_table has {n} rows; one row for each of the three routes gives at least 3",
    "table_no_times": "{step}: build_table has no column of times (numbers above zero)",
    "table_no_routes": "{step}: build_table does not name its three routes in a text column",
    "table_unnamed": "{step}: build_table does not say which row is which route (from tensors, from objects on the CPU then moved, from objects directly on the GPU)",
    "order_note": "{step}: unexpected order. Usually tensors < CPU then moved < directly on the GPU. Time the build only, after a warm-up run, and compare again.",
}


def line(key: str, **fields) -> str:
    """One check line, filled in."""

    return CHECK_LINES[key].format(**fields)
