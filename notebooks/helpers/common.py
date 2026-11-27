# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small setup helpers shared by the Core playbook modules."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_RUNTIME_ROOT = _PROJECT_ROOT / ".alchemi-runtime"
_DIRECT_KERNEL_CACHE_PATHS = {
    "AIMNET_CACHE_DIR": _LOCAL_RUNTIME_ROOT / "aimnet",
    "ALCHEMI_D3_PARAM_FILE": _LOCAL_RUNTIME_ROOT / "dftd3" / "dftd3_parameters.pt",
    "HF_HOME": _LOCAL_RUNTIME_ROOT / "huggingface",
    "MPLCONFIGDIR": _LOCAL_RUNTIME_ROOT / "matplotlib",
    "TORCH_HOME": _LOCAL_RUNTIME_ROOT / "torch",
    "TORCHINDUCTOR_CACHE_DIR": _LOCAL_RUNTIME_ROOT / "torch-inductor",
    "TRITON_CACHE_DIR": _LOCAL_RUNTIME_ROOT / "triton",
    "WARP_CACHE_PATH": _LOCAL_RUNTIME_ROOT / "warp",
}


def _configure_local_runtime() -> None:
    """Set cache fallbacks that still apply in a directly selected kernel."""

    for variable, path in _DIRECT_KERNEL_CACHE_PATHS.items():
        # The D3 tables are a prepared file: name it only where it exists, so
        # the lookup beside the Python runtime still applies elsewhere.
        if variable == "ALCHEMI_D3_PARAM_FILE" and not path.is_file():
            continue
        os.environ.setdefault(variable, str(path))


def _prime_tree_set_notice() -> None:
    """Consume dm-tree's one-time set notice outside learner-facing cells."""

    import tree

    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 2)
            tree.flatten({"energy", "forces"})
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)


def _ignore_known_aimnet2_cutoff_warning() -> None:
    """Ignore the pinned wrapper's scalar cutoff conversion notice.

    Toolkit's ``nvalchemi.models.base`` runs ``warnings.simplefilter("once", UserWarning)``
    when it is first imported, which puts that rule in front of every filter set
    before it. Import it first, so this filter lands in front of that rule.
    """

    try:
        import nvalchemi.models.base  # noqa: F401
    except Exception:  # without Toolkit there is no such rule to get in front of
        pass
    warnings.filterwarnings(
        "ignore",
        message="Converting a tensor with requires_grad=True to a scalar.*",
        category=UserWarning,
        module=r"nvalchemi\.models\.aimnet2",
    )


def register_course_fonts() -> list[str]:
    """Register the fonts shipped in ``shared/fonts`` with Matplotlib; return their family names.

    Matplotlib only finds fonts it has registered, so this runs before a course style
    names one (the figure standard uses Open Sans). A missing folder registers nothing.
    """

    from matplotlib import font_manager

    families = []
    folder = _PROJECT_ROOT / "shared" / "fonts"
    for path in sorted(folder.glob("*.ttf")) + sorted(folder.glob("*.otf")):
        font_manager.fontManager.addfont(str(path))
        families.append(font_manager.FontProperties(fname=str(path)).get_name())
    return families


def current_theme() -> str:
    """Return the course theme. The notebooks are light only."""

    return "light"


def configure_tutorial() -> None:
    """Apply the shared light tutorial style and silence known library notices.

    The course is light only: plots, widgets, the picker and the player use the
    light style.
    """

    _configure_local_runtime()
    import matplotlib.pyplot as plt

    _ignore_known_aimnet2_cutoff_warning()
    warnings.filterwarnings(
        "ignore",
        message=r'install "ipywidgets".*',
        category=UserWarning,
        module=r"rich\.live",
    )
    register_course_fonts()
    plt.style.use(_PROJECT_ROOT / "shared" / "alchemi-deck.mplstyle")
    _prime_tree_set_notice()


def tutorial_device():
    """Choose the workshop device and report the CPU fallback once."""

    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Compute device: {device} · {torch.cuda.get_device_name(device)}")
        return device
    print("Warning: CUDA is unavailable. This walkthrough is running on CPU.")
    return torch.device("cpu")
