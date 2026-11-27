"""Write the liquid SiC cut of notebook 09 from the SiC-TGAP training set.

Source: dataset-SiC-TGAP.xyz of the Zenodo record 10.5281/zenodo.17182227 (CC BY 4.0; see README.md and
cut/LICENSE). The file is not in this repository: download it and save it as raw/dataset-SiC-TGAP.xyz beside this
script. Writes cut/:

  liquid-train.extxyz.gz     300 frames of 216 atoms (Si108C108): every frame of the four liquid frame types
                             (melt_aimd, eqlMelt_aimd, quench_aimd, amorph_aimd) at the five training densities
                             2.3, 2.5, 2.9, 3.05 and 3.2 g/cm3; the starts of the notebook's MD are its
                             eqlMelt_aimd frames
  liquid-heldout.extxyz.gz   60 frames: the whole density 2.7 lineage of the same four types, never trained on
  cut-record.json            source frame indices, counts by type and density, SHA-256 of the source and the cut

Each frame keeps its DFT labels as dft_energy (eV, frame info) and dft_forces (eV/Å, per atom), with config_type,
sub_config and source_index (0-based frame index in the source). The source's energy, forces, virial and stress
fields are not copied. The frames keep the order of the source within each frame type. The output has no time
stamp: every run on the same source gives the same bytes.

Usage, from this folder:  python cut_liquid.py
"""

import collections
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from ase.io import iread, write

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw" / "dataset-SiC-TGAP.xyz"
OUT = HERE / "cut"
TYPES = ["melt_aimd", "eqlMelt_aimd", "quench_aimd", "amorph_aimd"]
HELD_OUT_DENSITY = 2.7


def density(sub_config: str) -> float:
    """g/cm3 from a sub_config: "rho2.7", "rho2.7_Tm3000", "iter1_rho_2.7_Tm_3000", "iter1_rho_3.053_Tm_3700"."""

    text = sub_config.replace("rho_", "rho").split("rho")[1].split("_")[0]
    return 3.05 if text == "3.053" else round(float(text), 2)


def clean(index: int, atoms):
    """A copy with the DFT labels under their own names and nothing else."""

    copy = atoms.copy()
    energy, forces = atoms.get_potential_energy(), atoms.get_forces()
    copy.calc = None
    copy.info = {"dft_energy": float(energy), "config_type": atoms.info.get("config_type"),
                 "sub_config": atoms.info.get("sub_config", ""), "source_index": index}
    for key in list(copy.arrays):
        if key not in ("numbers", "positions"):
            del copy.arrays[key]
    copy.arrays["dft_forces"] = np.asarray(forces)
    return copy


def write_gz(path: Path, frames) -> str:
    """Write extxyz through gzip with no name or time in the header (same bytes on every run); SHA-256 of the text."""

    text = io.StringIO()
    write(text, frames, format="extxyz")
    data = text.getvalue().encode()
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
        gz.write(data)
    return hashlib.sha256(data).hexdigest()


def select(frames: dict, held_out: bool) -> list[int]:
    """Source indices of the liquid frames at the held-out density, or at the others: by type, in source order."""

    return [i for t in TYPES for i in sorted(frames)
            if frames[i].info["config_type"] == t
            and (density(frames[i].info["sub_config"]) == HELD_OUT_DENSITY) == held_out]


def summary(frames: dict, indices: list[int]) -> dict:
    """The record of one file: frame count, source indices, counts by type and by density."""

    by_density = collections.Counter(density(frames[i].info["sub_config"]) for i in indices)
    return {"frames": len(indices), "source_index": indices,
            "by_type": dict(collections.Counter(frames[i].info["config_type"] for i in indices)),
            "by_density": {str(k): v for k, v in sorted(by_density.items())}}


def main() -> None:
    frames = {i: a for i, a in enumerate(iread(RAW, index=":", format="extxyz")) if a.info.get("config_type") in TYPES}
    train_idx, held_idx = select(frames, held_out=False), select(frames, held_out=True)
    assert not set(train_idx) & set(held_idx)
    assert all(len(frames[i]) == 216 for i in train_idx + held_idx)
    OUT.mkdir(exist_ok=True)
    train_sha = write_gz(OUT / "liquid-train.extxyz.gz", [clean(i, frames[i]) for i in train_idx])
    held_sha = write_gz(OUT / "liquid-heldout.extxyz.gz", [clean(i, frames[i]) for i in held_idx])
    record = {
        "source": {"file": "raw/dataset-SiC-TGAP.xyz", "record": "https://zenodo.org/records/17182227",
                   "doi": "10.5281/zenodo.17182227", "licence": "CC BY 4.0",
                   "sha256": hashlib.sha256(RAW.read_bytes()).hexdigest()},
        "rule": "every frame of the four liquid types; density 2.7 held out whole",
        "train": summary(frames, train_idx),
        "held_out": summary(frames, held_idx),
        "sha256": {"liquid-train.extxyz (uncompressed)": train_sha, "liquid-heldout.extxyz (uncompressed)": held_sha},
    }
    (OUT / "cut-record.json").write_text(json.dumps(record, indent=1) + "\n")
    for name in ("train", "held_out"):  # the counts, without the long index lists
        print(name, json.dumps({k: v for k, v in record[name].items() if k != "source_index"}))
    print("sha256", json.dumps(record["sha256"], indent=1))


if __name__ == "__main__":
    main()
