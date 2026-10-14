"""Loader for the DrivAerNet++ surface pressure dataset (VTK meshes).

Adapted from ``denoising-diffusion-pytorch-fluid-mechanics/data_generation/driveaernet_data.py``.

Usage:
    dataset_train, dataset_valid, dataset_test, coefficients = load_drivaernet(
        cache_path, vtk_dir=vtk_dir, split_dir=split_dir)

``vtk_dir`` holds the folders of ``.vtk`` files (one car each, with the point
field ``p``) and ``split_dir`` the ``train.txt``, ``validation.txt`` and
``test.txt`` files listing the car names. The meshes are read once and cached
(in physical units) to ``cache_path``; later runs only need the cache.

Each car has a different number of surface points (~360k-560k), so the
datasets return unpadded samples and :func:`drivaernet_collate_fn` pads them
per batch and builds the point mask.
"""

import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


SPLIT_FILES = {"train": "train.txt", "valid": "validation.txt", "test": "test.txt"}


def _spread_bits(v):
    """Insert two zero bits after each of the 21 lowest bits of v (uint64)."""
    v = v & np.uint64(0x1FFFFF)
    for shift, mask in ((32, 0x1F00000000FFFF), (16, 0x1F0000FF0000FF), (8, 0x100F00F00F00F00F),
                        (4, 0x10C30C30C30C30C3), (2, 0x1249249249249249)):
        v = (v | (v << np.uint64(shift))) & np.uint64(mask)
    return v


def morton_order(xyz):
    """Permutation sorting the points (N, 3) along a Morton (Z-order) curve.

    Consecutive points in the VTK files are scattered over the whole car
    (a patch of 8 of them spans ~1.6 m), so patches of consecutive points would
    mix unrelated regions. After sorting, a patch of 16 points spans a few cm.
    The order only depends on the points, so it can be recomputed from the VTK
    to map predictions back: ``values_vtk[order] = values_sorted``.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    lo = xyz.min(axis=0)
    extent = max((xyz.max(axis=0) - lo).max(), 1e-12)
    q = ((xyz - lo) / extent * (2**21 - 1)).astype(np.uint64)
    code = np.zeros(len(xyz), dtype=np.uint64)
    for axis in range(3):
        code |= _spread_bits(q[:, axis]) << np.uint64(axis)
    return np.argsort(code, kind="stable")


def _read_vtk(args):
    """(path, field_names, sort_points) -> xyz (N, 3), fields (N, C) as float32."""
    import pyvista as pv
    path, field_names, sort_points = args
    mesh = pv.read(path)
    xyz = np.asarray(mesh.points, dtype=np.float32)
    # vector fields (e.g. wall shear stress) contribute one channel per component
    fields = np.concatenate(
        [np.asarray(mesh.point_data[name], dtype=np.float32).reshape(len(xyz), -1) for name in field_names],
        axis=1)
    if sort_points:
        order = morton_order(xyz)
        xyz, fields = xyz[order], fields[order]
    return xyz, fields


def index_vtk_files(vtk_dir):
    """car name -> .vtk path, over the subfolders of vtk_dir."""
    paths = {}
    for folder in sorted(os.listdir(vtk_dir)):
        folder_path = os.path.join(vtk_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            name, ext = os.path.splitext(fname)
            if ext == ".vtk":
                paths.setdefault(name, os.path.join(folder_path, fname))
    return paths


def _mean_std(x, chunk=2**25):
    """Per-column mean / std of a (N, d) tensor, accumulated in float64.

    A float32 ``mean(0)`` over billions of points is badly off (the old cache
    had a coordinate mean of 0.04 for a car spanning -0.7..4.0 m).
    """
    s1 = torch.zeros(x.shape[1], dtype=torch.float64)
    s2 = torch.zeros(x.shape[1], dtype=torch.float64)
    for i in range(0, len(x), chunk):
        c = x[i:i + chunk].double()
        s1 += c.sum(0)
        s2 += (c ** 2).sum(0)
    mean = s1 / len(x)
    std = (s2 / len(x) - mean ** 2).clamp(min=0).sqrt()
    return mean.float(), std.float()


def build_drivaernet_cache(cache_path, vtk_dir, split_dir, field_names=("p",), sort_points=True,
                           num_workers=16):
    """Read the VTK meshes of every split and save them to ``cache_path``.

    Each split is stored flat: ``coords`` (sum N_i, 3), ``fields`` (sum N_i, C)
    in physical units and ``offsets`` (n + 1,), so the cache loads instantly
    with ``mmap``. The normalization stats are computed on the train split.
    """
    paths = index_vtk_files(vtk_dir)
    cache = {"field_names": list(field_names), "sort_points": sort_points}
    for split, split_file in SPLIT_FILES.items():
        with open(os.path.join(split_dir, split_file)) as f:
            names = [line.strip() for line in f if line.strip()]
        missing = [n for n in names if n not in paths]
        if missing:
            print(f"[{split}] {len(missing)} cars without a .vtk file, skipped: "
                  f"{missing[:3]}{'...' if len(missing) > 3 else ''}")
        names = [n for n in names if n in paths]
        print(f"[{split}] reading {len(names)} meshes with {num_workers} processes...")
        coords, fields = [], []
        args = [(paths[n], tuple(field_names), sort_points) for n in names]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for xyz, f in tqdm(executor.map(_read_vtk, args, chunksize=4), total=len(args)):
                coords.append(xyz)
                fields.append(f)
        cache[split] = {
            "names": names,
            "coords": torch.from_numpy(np.concatenate(coords)),
            "fields": torch.from_numpy(np.concatenate(fields)),
            "offsets": torch.from_numpy(np.cumsum([0] + [len(c) for c in coords])),
        }
        del coords, fields

    coords_mean, coords_std = _mean_std(cache["train"]["coords"])
    fields_mean, fields_std = _mean_std(cache["train"]["fields"])
    cache["stats"] = {
        "coords_mean": coords_mean, "coords_std": coords_std,
        "fields_mean": fields_mean, "fields_std": fields_std,
    }
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    torch.save(cache, cache_path)
    print(f"Saved DrivAerNet cache to {cache_path}")


class DrivAerNetDataset(Dataset):
    """Cars of one split, returned unpadded and normalized.

    Items are ``(fields (C, N_i), coords (3, N_i))``; use
    :func:`drivaernet_collate_fn` to batch them.
    """
    def __init__(self, split_data, stats):
        self.names = split_data["names"]
        self.coords = split_data["coords"]
        self.fields = split_data["fields"]
        self.offsets = split_data["offsets"]
        self.stats = stats

    def __len__(self):
        return len(self.names)

    def num_points(self, idx):
        return int(self.offsets[idx + 1] - self.offsets[idx])

    def __getitem__(self, idx):
        start, end = int(self.offsets[idx]), int(self.offsets[idx + 1])
        s = self.stats
        coords = (self.coords[start:end] - s["coords_mean"]) / s["coords_std"]
        fields = (self.fields[start:end] - s["fields_mean"]) / s["fields_std"]
        return fields.T.contiguous(), coords.T.contiguous()


def drivaernet_collate_fn(batch, pad_multiple=1):
    """Pad a list of ``(fields (C, N_i), coords (3, N_i))`` to a common length.

    The length is rounded up to a multiple of ``pad_multiple`` (a multiple of
    the model patch size). Returns a dict with the target ``x`` (B, C, L), the
    model input ``context`` (B, 3, L) and the point ``mask`` (B, L), True on
    real points; padded entries are zero.
    """
    fields, coords = zip(*batch)
    lengths = torch.tensor([f.shape[-1] for f in fields])
    length = math.ceil(lengths.max().item() / pad_multiple) * pad_multiple
    x = fields[0].new_zeros(len(batch), fields[0].shape[0], length)
    context = coords[0].new_zeros(len(batch), coords[0].shape[0], length)
    for i, (f, c) in enumerate(batch):
        x[i, :, :f.shape[-1]] = f
        context[i, :, :c.shape[-1]] = c
    mask = torch.arange(length) < lengths[:, None]
    return {"x": x, "context": context, "mask": mask}


def load_drivaernet(cache_path, vtk_dir=None, split_dir=None, field_names=("p",), sort_points=True,
                    num_workers=16, mmap=True):
    """Load the DrivAerNet++ train / validation / test splits.

    Args:
        cache_path: cache file; built from ``vtk_dir`` / ``split_dir`` if missing.
        vtk_dir: folder with the subfolders of ``.vtk`` files.
        split_dir: folder with ``train.txt``, ``validation.txt`` and ``test.txt``.
        field_names: VTK point fields to predict (the cache must match).
        sort_points: sort the points of each car along a Morton curve, so
            that consecutive points (one model patch) are spatial neighbours.
        num_workers: processes used to read the VTK files.
        mmap: memory-map the cache instead of reading it all into RAM.

    Returns:
        dataset_train, dataset_valid, dataset_test, coefficients, where
        coefficients holds the normalization stats (coords / fields mean and
        std, in physical units) and the field names.
    """
    if not os.path.exists(cache_path):
        if vtk_dir is None or split_dir is None:
            raise FileNotFoundError(f"No cache at {cache_path}: pass vtk_dir and split_dir to build it")
        build_drivaernet_cache(cache_path, vtk_dir, split_dir, field_names=field_names,
                               sort_points=sort_points, num_workers=num_workers)

    print(f"Loading DrivAerNet cache {cache_path} (mmap={mmap})...")
    cache = torch.load(cache_path, mmap=mmap, weights_only=True)
    if cache["field_names"] != list(field_names) or cache["sort_points"] != sort_points:
        raise ValueError(
            f"{cache_path} holds fields {cache['field_names']} with sort_points={cache['sort_points']}, "
            f"but fields {list(field_names)} with sort_points={sort_points} were requested: "
            "use another cache_path to build a new cache")

    stats = cache["stats"]
    datasets = [DrivAerNetDataset(cache[split], stats) for split in SPLIT_FILES]
    for split, ds in zip(SPLIT_FILES, datasets):
        n = ds.offsets.diff()
        print(f"[{split}] {len(ds)} cars, points per car min {n.min().item()} "
              f"mean {n.float().mean().item():.0f} max {n.max().item()}")
    print("coords mean", stats["coords_mean"].tolist(), "std", stats["coords_std"].tolist())
    print("fields", list(field_names), "mean", stats["fields_mean"].tolist(), "std", stats["fields_std"].tolist())

    coefficients = {**stats, "field_names": list(field_names)}
    return (*datasets, coefficients)


def write_prediction_vtk(vtk_path, out_path, pred, field_names=("p",), sort_points=True):
    """Save a car mesh with the predicted fields and their errors, for ParaView.

    pred: (C, N) prediction in physical units, in the dataset point order.
    Adds ``<name>_pred`` and ``<name>_error`` (pred - true) point fields.
    """
    import pyvista as pv
    mesh = pv.read(vtk_path)
    pred = np.asarray(pred, dtype=np.float32).T  # (N, C)
    if sort_points:
        unsorted = np.empty_like(pred)
        unsorted[morton_order(mesh.points)] = pred
        pred = unsorted
    channel = 0
    for name in field_names:
        true = np.asarray(mesh.point_data[name], dtype=np.float32)
        n_comp = true.size // mesh.n_points
        p = pred[:, channel:channel + n_comp].reshape(true.shape)
        mesh.point_data[f"{name}_pred"] = p
        mesh.point_data[f"{name}_error"] = p - true
        channel += n_comp
    mesh.save(out_path)
