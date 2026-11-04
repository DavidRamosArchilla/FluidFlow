"""Loader for the flow-past-a-cylinder dataset (HDF5, one file per simulated case).

Layout of a split folder (e.g. ``data/cylinder/train``)::

    constants.h5    shared by every case of the split
        vertices (V, 2)    mesh nodes
        indices  (L, 4)    quad cells (vertex ids)
        centers  (L, 2)    cell centroids, where the fields live
        cases/<key> (N,)   per-case parameters, e.g. cases/Re
        attrs Lc, Uc       reference length (cylinder diameter) and velocity
    <i>.h5          case i (parameters cases/<key>[i])
        fields  (T, 3, L)  [p, u, v] per cell
        scalars (T, 2)     [Re, time since the previous snapshot]

The channel order and the meaning of the scalars are not stored in the files;
see ``data/cylinder/check_cylinder_dataset.ipynb`` for how they were identified.

Usage:
    dataset_train, dataset_valid, dataset_test, coefficients = load_cylinder(data_dir)

The datasets are ``TensorDataset(fields, conds)`` with fields (N, F, C, L) in
the [u, v, p] order of the airfoil loader, so the autoregressive helpers of
``data/load_airfoil_unsteady.py`` (make_pair_dataset, make_rollout_dataset,
...) apply directly.
"""

import math
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset


FILE_CHANNELS = ["p", "u", "v"]   # order in the h5 files
CHANNEL_NAMES = ["u", "v", "p"]   # order of the returned fields (as in the airfoil loader)
CHANNEL_ORDER = [FILE_CHANNELS.index(name) for name in CHANNEL_NAMES]


def load_cylinder_mesh(split_dir):
    """Mesh of a split: dict with vertices (V, 2), cells (L, 4), centers (L, 2), Lc, Uc."""
    with h5py.File(os.path.join(split_dir, "constants.h5"), "r") as f:
        return {
            "vertices": f["vertices"][:],
            "cells": f["indices"][:],
            "centers": f["centers"][:].astype(np.float32),
            "Lc": float(f.attrs["Lc"]),
            "Uc": float(f.attrs["Uc"]),
        }


def load_cylinder_split(split_dir, cond_keys=("Re",), time_stride=1, max_samples=None, dtype=np.float32):
    """Load every case of one split.

    Args:
        split_dir: folder with ``constants.h5`` and the ``<i>.h5`` case files.
        cond_keys: datasets under ``cases/`` in constants.h5 used as conditions.
        time_stride: keep every k-th snapshot (larger autoregressive step).
        max_samples: load only the first cases (None = all).

    Returns:
        fields (N, F, 3, L) as [u, v, p], conds (N, len(cond_keys)), times (N, F)
        (cumulative time of each snapshot). Trajectories of different length
        are cut to the shortest one.
    """
    with h5py.File(os.path.join(split_dir, "constants.h5"), "r") as f:
        params = {k: f[f"cases/{k}"][:] for k in f["cases"]}
    missing = [k for k in cond_keys if k not in params]
    assert not missing, f"{split_dir}: no cases/{missing} in constants.h5 (has {list(params)})"
    n_cases = len(next(iter(params.values())))
    if max_samples is not None:
        n_cases = min(n_cases, max_samples)

    fields_list, times_list = [], []
    for i in range(n_cases):
        case_path = os.path.join(split_dir, f"{i}.h5")
        with h5py.File(case_path, "r") as case_file:
            scalars = case_file["scalars"][:]
            fields = case_file["fields"][:]
        if "Re" in params:
            assert np.allclose(scalars[:, 0], params["Re"][i]), \
                f"{case_path} has Re {np.unique(scalars[:, 0])}, constants.h5 cases/Re[{i}] = {params['Re'][i]}"
        fields_list.append(fields[::time_stride, CHANNEL_ORDER].astype(dtype))
        times_list.append(np.cumsum(scalars[:, 1])[::time_stride])
        if (i + 1) % 50 == 0:
            print(f"  loaded {i + 1}/{n_cases} cases from {split_dir}...")
    if n_cases == 0:
        raise RuntimeError(f"No cases found in {split_dir}")

    n_frames = min(len(t) for t in times_list)
    if any(len(t) != n_frames for t in times_list):
        print(f"[{split_dir}] trajectories have {sorted({len(t) for t in times_list})} frames, cutting all to {n_frames}")
    fields = torch.from_numpy(np.stack([f[:n_frames] for f in fields_list]))
    times = torch.from_numpy(np.stack([t[:n_frames] for t in times_list]))
    conds = torch.from_numpy(np.stack([params[k][:n_cases] for k in cond_keys], axis=1).astype(np.float32))
    print(f"[{split_dir}] fields {tuple(fields.shape)} (N,F,C,L), conds {tuple(conds.shape)} {list(cond_keys)} "
          f"in [{conds.min(0).values.tolist()}, {conds.max(0).values.tolist()}]")
    return fields, conds, times


def load_cylinder(data_dir, cond_keys=("Re",), time_stride=1, max_train_samples=None,
                  max_valid_samples=None, max_test_samples=None, patch_size=None):
    """Load the cylinder dataset.

    Args:
        data_dir: folder with ``train/`` and ``test/`` (and optionally ``valid/``;
            without it the test split is used for validation).
        cond_keys: datasets under ``cases/`` used as conditions (e.g. ("Re",)).
        time_stride: keep every k-th snapshot.
        max_train_samples / max_valid_samples / max_test_samples: cap per split (None = all).
        patch_size: if given, pad the cell dimension to a multiple of it.

    Returns:
        dataset_train, dataset_valid, dataset_test, coefficients, where coefficients
        holds the normalization stats (fields/conds mean and std), the padding
        info (target_length, original_length), the mesh (centers, cells,
        vertices), channel_names, cond_names and dt (median time between kept frames).
    """
    print("Loading cylinder data...")
    split_dirs = {s: os.path.join(data_dir, s) for s in ("train", "valid", "test")}
    if not os.path.isdir(split_dirs["valid"]):
        print(f"No {split_dirs['valid']}, validating on the test split")
        split_dirs["valid"] = split_dirs["test"]

    mesh = load_cylinder_mesh(split_dirs["train"])
    for split in ("valid", "test"):
        other = load_cylinder_mesh(split_dirs[split])
        assert np.array_equal(other["cells"], mesh["cells"]) and np.allclose(other["centers"], mesh["centers"]), \
            f"the {split} mesh differs from the train mesh"

    fields_train, conds_train, times_train = load_cylinder_split(
        split_dirs["train"], cond_keys, time_stride, max_train_samples)
    fields_valid, conds_valid, _ = load_cylinder_split(split_dirs["valid"], cond_keys, time_stride, max_valid_samples)
    fields_test, conds_test, _ = load_cylinder_split(split_dirs["test"], cond_keys, time_stride, max_test_samples)
    dt = float(torch.diff(times_train, dim=1).median())

    # standardize per channel with the train stats
    fields_mean = fields_train.mean(dim=(0, 1, 3), keepdim=True)
    fields_std = fields_train.std(dim=(0, 1, 3), keepdim=True)
    print("fields_train mean per channel", dict(zip(CHANNEL_NAMES, fields_mean.flatten().tolist())))
    print("fields_train std per channel", dict(zip(CHANNEL_NAMES, fields_std.flatten().tolist())))
    fields_train = (fields_train - fields_mean) / fields_std
    fields_valid = (fields_valid - fields_mean) / fields_std
    fields_test = (fields_test - fields_mean) / fields_std

    # biased std: with a single training case (or a single Re) it is 0 instead of NaN
    conds_mean = conds_train.mean(dim=0, keepdim=True)
    conds_std = conds_train.std(dim=0, correction=0, keepdim=True).clamp(min=1e-6)
    print("conds_train mean", conds_mean.tolist(), "std", conds_std.tolist())
    conds_train = (conds_train - conds_mean) / conds_std
    conds_valid = (conds_valid - conds_mean) / conds_std
    conds_test = (conds_test - conds_mean) / conds_std

    # pad the cell dimension to a multiple of patch_size (if requested)
    original_length = fields_train.shape[-1]
    target_length = original_length
    if patch_size is not None:
        target_length = math.ceil(original_length / patch_size) * patch_size
        pad_length = target_length - original_length
        if pad_length:
            print(f"Padding cells {original_length} -> {target_length} (patch {patch_size} pad {pad_length})")
            fields_train = F.pad(fields_train, (0, pad_length))
            fields_valid = F.pad(fields_valid, (0, pad_length))
            fields_test = F.pad(fields_test, (0, pad_length))

    coefficients = {
        "fields_mean": fields_mean,
        "fields_std": fields_std,
        "conds_mean": conds_mean,
        "conds_std": conds_std,
        "target_length": target_length,
        "original_length": original_length,
        "centers": mesh["centers"],
        "cells": mesh["cells"],
        "vertices": mesh["vertices"],
        "channel_names": CHANNEL_NAMES,
        "cond_names": list(cond_keys),
        "dt": dt,
    }
    return (TensorDataset(fields_train, conds_train), TensorDataset(fields_valid, conds_valid),
            TensorDataset(fields_test, conds_test), coefficients)
