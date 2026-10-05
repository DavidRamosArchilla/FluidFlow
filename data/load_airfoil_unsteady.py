"""Loader for the unsteady airfoil dataset (MeshGraphNets TFRecords).

Adapted from the data-loading logic in
``denoising-diffusion-pytorch-fluid-mechanics/scripts/train_airfoil_unsteady.py``.

Usage:
    dataset_train, dataset_valid, dataset_test, coefficients = load_airfoil_unsteady(data_dir)

``data_dir`` is the folder containing ``meta.json`` + ``train.tfrecord``,
``valid.tfrecord`` and ``test.tfrecord`` (see ``data/airfoil_unsteady/download_data.sh``).
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset


T_FULL = 601  # number of timesteps per trajectory in the raw files


def load_meta(meta_path):
    with open(meta_path) as f:
        return json.load(f)


def decode_example(raw, meta):
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    ex = tf.train.Example()
    ex.ParseFromString(raw.numpy())
    out = {}
    for k, v in meta["features"].items():
        f = ex.features.feature[k]
        dtype = np.dtype(v["dtype"].replace("int32", "<i4").replace("float32", "<f4"))
        arr = np.frombuffer(f.bytes_list.value[0], dtype=dtype)
        arr = arr.reshape(v["shape"])
        out[k] = arr
    return out


def extract_conditions(velocity, density, pressure, node_type, gamma=1.4):
    far_mask = node_type[0, :, 0] == 4
    u_inf = velocity[0, far_mask, 0].mean()
    v_inf = velocity[0, far_mask, 1].mean()
    rho_inf = density[0, far_mask, 0].mean()
    p_inf = pressure[0, far_mask, 0].mean()
    mag = np.sqrt(u_inf**2 + v_inf**2)
    a_inf = np.sqrt(gamma * p_inf / rho_inf)
    mach = mag / a_inf
    alpha = np.degrees(np.arctan2(v_inf, u_inf))
    return mach, alpha


def load_airfoil_split(split="train", base_dir=None, meta_path=None, channels=3,
                       time_frames=None, time_mode="uniform", max_samples=None,
                       dtype=np.float32, return_node_type=False):
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    if meta_path is None:
        meta_path = os.path.join(base_dir, "meta.json")
    meta = load_meta(meta_path)
    tfrecord_path = os.path.join(base_dir, f"{split}.tfrecord")
    assert os.path.exists(tfrecord_path), f"not found {tfrecord_path}"
    ds = tf.data.TFRecordDataset(tfrecord_path)
    data_list = []
    cond_list = []
    mesh_pos_ref = None
    cells_ref = None
    node_type_ref = None
    if time_frames is None or time_frames >= T_FULL:
        time_idx = None
    else:
        if time_mode == "uniform":
            time_idx = np.linspace(0, T_FULL - 1, time_frames, dtype=int)
        elif time_mode == "first":
            time_idx = np.arange(time_frames)
        else:
            raise ValueError(time_mode)
        print(f"[{split}] temporal subsampling: {time_mode} {T_FULL}->{time_frames} idx {time_idx[:5]}...{time_idx[-5:]}")
    count = 0
    for raw in ds:
        if max_samples is not None and count >= max_samples:
            break
        out = decode_example(raw, meta)
        vel = out["velocity"]
        press = out["pressure"]
        dens = out["density"]
        node_type = out["node_type"]
        if mesh_pos_ref is None:
            mesh_pos_ref = out["mesh_pos"]
            cells_ref = out["cells"]
            node_type_ref = out["node_type"]
        if time_idx is not None:
            vel = vel[time_idx]
            press = press[time_idx]
            dens = dens[time_idx]
        vel_t = np.transpose(vel, (0, 2, 1))
        press_t = np.transpose(press, (0, 2, 1))
        dens_t = np.transpose(dens, (0, 2, 1))
        if channels == 3:
            stacked = np.concatenate([vel_t, press_t], axis=1)
        elif channels == 4:
            stacked = np.concatenate([vel_t, press_t, dens_t], axis=1)
        else:
            raise ValueError("channels 3 or 4")
        stacked = stacked.astype(dtype)
        data_list.append(stacked)
        mach, alpha = extract_conditions(out["velocity"], out["density"], out["pressure"], out["node_type"])
        cond_list.append([mach, alpha])
        count += 1
        if count % 10 == 0:
            print(f"  loaded {count} samples for {split}...")
    if len(data_list) == 0:
        raise RuntimeError(f"No data loaded for split {split}")
    data_np = np.stack(data_list, axis=0)
    cond_np = np.array(cond_list, dtype=np.float32)
    data_torch = torch.from_numpy(data_np)
    cond_torch = torch.from_numpy(cond_np)
    print(f"[{split}] data shape {data_torch.shape} (N,F,C,Nnodes) dtype {data_torch.dtype}")
    print(f"[{split}] cond shape {cond_torch.shape} Mach [{cond_torch[:, 0].min():.3f},{cond_torch[:, 0].max():.3f}] Alpha [{cond_torch[:, 1].min():.2f},{cond_torch[:, 1].max():.2f}]")
    print(f"  mesh_pos {mesh_pos_ref.shape} cells {cells_ref.shape}")
    if return_node_type:
        # (1, L, 1) -> (L,) node types: 0 interior, 2 wall, 4 farfield
        return data_torch, cond_torch, mesh_pos_ref, cells_ref, node_type_ref[0, :, 0]
    return data_torch, cond_torch, mesh_pos_ref, cells_ref


def load_airfoil_unsteady(data_dir, channels=3, time_frames=None, time_mode="uniform",
                          max_train_samples=None, max_valid_samples=None,
                          max_test_samples=None, patch_size=None):
    """Load the unsteady airfoil dataset.

    Args:
        data_dir: folder with ``meta.json`` + train/valid/test ``.tfrecord`` files.
        channels: 3 -> [u, v, p], 4 -> [u, v, p, rho].
        time_frames: temporal subsampling (None for the full 601 frames).
        time_mode: "uniform" or "first".
        max_train_samples / max_valid_samples / max_test_samples: cap per split (None = all).
        patch_size: if given, pad the node dimension to a multiple of it.

    Returns:
        dataset_train, dataset_valid, dataset_test, coefficients, where coefficients
        holds the normalization stats (fields/conds mean and std), the
        padding info (target_length, original_length) and the mesh
        (mesh_pos, cells, node_type), like ``load_onera_crm``.
    """
    print("Loading airfoil unsteady data...")
    fields_train, conds_train, mesh_pos, cells, node_type = load_airfoil_split(
        split="train", base_dir=data_dir, channels=channels,
        time_frames=time_frames, time_mode=time_mode, max_samples=max_train_samples,
        return_node_type=True)
    fields_valid, conds_valid, _, _ = load_airfoil_split(
        split="valid", base_dir=data_dir, channels=channels,
        time_frames=time_frames, time_mode=time_mode, max_samples=max_valid_samples)
    fields_test, conds_test, _, _ = load_airfoil_split(
        split="test", base_dir=data_dir, channels=channels,
        time_frames=time_frames, time_mode=time_mode, max_samples=max_test_samples)

    print("Raw fields_train", fields_train.shape, "fields_valid", fields_valid.shape, "fields_test", fields_test.shape)
    print("Raw conds_train", conds_train.shape)

    # standardize the data per field independently (train stats applied to all splits)
    fields_train_mean = fields_train.mean(dim=(0, 1, 3), keepdim=True)
    fields_train_std = fields_train.std(dim=(0, 1, 3), keepdim=True)
    print("fields_train mean per channel", fields_train_mean.squeeze().tolist())
    print("fields_train std per channel", fields_train_std.squeeze().tolist())
    fields_train = (fields_train - fields_train_mean) / fields_train_std
    fields_valid = (fields_valid - fields_train_mean) / fields_train_std
    fields_test = (fields_test - fields_train_mean) / fields_train_std

    conds_mean = conds_train.mean(dim=0, keepdim=True)
    conds_std = torch.clamp(conds_train.std(dim=0, keepdim=True), min=1e-6)
    print("conds_train mean", conds_mean, "std", conds_std)
    conds_train = (conds_train - conds_mean) / conds_std
    conds_valid = (conds_valid - conds_mean) / conds_std
    conds_test = (conds_test - conds_mean) / conds_std
    print("conds_train normalized", conds_train.shape, conds_train[0])
    print("fields_train normalized mean", fields_train.mean(dim=(0, 1, 3)).squeeze())
    print("fields_train normalized std", fields_train.std(dim=(0, 1, 3)).squeeze())

    # pad the node dimension to a multiple of patch_size (if requested)
    original_length = fields_train.shape[-1]
    if patch_size is not None:
        target_length = math.ceil(original_length / patch_size) * patch_size
        pad_length = target_length - original_length
        print("fields_train shape", fields_train.shape)
        print("fields_valid shape", fields_valid.shape)
        print("fields_test shape", fields_test.shape)
        print(f"Padding nodes {original_length} -> {target_length} (patch {patch_size} pad {pad_length})")
        fields_train = F.pad(fields_train, (0, pad_length))
        fields_valid = F.pad(fields_valid, (0, pad_length))
        fields_test = F.pad(fields_test, (0, pad_length))
        print("fields_train padded", fields_train.shape)
        print("fields_valid padded", fields_valid.shape)
        print("fields_test padded", fields_test.shape)
    else:
        target_length = original_length

    dataset_train = TensorDataset(fields_train, conds_train)
    dataset_valid = TensorDataset(fields_valid, conds_valid)
    dataset_test = TensorDataset(fields_test, conds_test)

    coefficients = {
        'fields_mean': fields_train_mean,
        'fields_std': fields_train_std,
        'conds_mean': conds_mean,
        'conds_std': conds_std,
        'target_length': target_length,
        'original_length': original_length,
        'mesh_pos': mesh_pos,
        'cells': cells,
        'node_type': node_type,
    }
    return dataset_train, dataset_valid, dataset_test, coefficients


# --------------------------------------------------------------------------
# Autoregressive (one-step) datasets
# --------------------------------------------------------------------------

class AutoregressivePairDataset(Dataset):
    """One-step pairs (u_{t+1}, conds, u_t) built lazily from (N, F, C, L) fields.

    Items are ordered as ``(target, conds, context)`` so the Trainer passes
    ``context`` (the current state) as model input and compares against the
    next state. No pair tensor is materialised, so memory stays at the size
    of ``fields``.

    Args:
        fields: (N, F, C, L) tensor (normalized fields or latents).
        conds: (N, cond_dim) tensor.
        frame_stride: keep every ``frame_stride``-th starting frame t
            (e.g. to subsample the validation pairs).
        max_pairs: optional cap on the number of pairs (evenly spaced).
    """
    def __init__(self, fields, conds, frame_stride=1, max_pairs=None):
        assert fields.ndim == 4 and fields.shape[1] > 1, "fields must be (N, F, C, L) with F > 1"
        assert fields.shape[0] == conds.shape[0]
        self.fields = fields
        self.conds = conds
        starts = torch.arange(0, fields.shape[1] - 1, frame_stride)
        sims = torch.arange(fields.shape[0])
        # (P, 2) list of (sim, t)
        self.index = torch.stack(torch.meshgrid(sims, starts, indexing="ij"), dim=-1).reshape(-1, 2)
        if max_pairs is not None and len(self.index) > max_pairs:
            keep = torch.linspace(0, len(self.index) - 1, max_pairs).long()
            self.index = self.index[keep]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        n, t = self.index[i].tolist()
        return self.fields[n, t + 1], self.conds[n], self.fields[n, t]


def make_pair_dataset(dataset, frame_stride=1, max_pairs=None):
    """(fields, conds) TensorDataset -> :class:`AutoregressivePairDataset`."""
    fields, conds = dataset.tensors[0], dataset.tensors[1]
    return AutoregressivePairDataset(fields, conds, frame_stride=frame_stride, max_pairs=max_pairs)


def make_rollout_dataset(dataset):
    """(fields, conds) TensorDataset -> (fields, conds, fields[:, 0]).

    Used to evaluate full autoregressive rollouts: the model receives the
    initial frame as ``context`` and the whole trajectory is the target.
    """
    fields, conds = dataset.tensors[0], dataset.tensors[1]
    return TensorDataset(fields, conds, fields[:, 0].contiguous())


@torch.no_grad()
def compute_delta_stats(fields, valid_length=None, chunk=16):
    """Per-channel mean/std of the one-step increments u_{t+1} - u_t.

    fields: (N, F, C, L). Computed in chunks over N (float64 accumulation) to
    avoid materialising the full difference tensor. Nodes beyond
    ``valid_length`` (padding) are ignored.

    Returns (mean, std), each of shape (C,).
    """
    n_ch = fields.shape[2]
    s1 = torch.zeros(n_ch, dtype=torch.float64)
    s2 = torch.zeros(n_ch, dtype=torch.float64)
    count = 0
    for i in range(0, fields.shape[0], chunk):
        x = fields[i:i + chunk]
        if valid_length is not None:
            x = x[..., :valid_length]
        d = (x[:, 1:] - x[:, :-1]).double()  # (n, F-1, C, L)
        s1 += d.sum(dim=(0, 1, 3))
        s2 += (d ** 2).sum(dim=(0, 1, 3))
        count += d.shape[0] * d.shape[1] * d.shape[3]
    mean = s1 / count
    std = (s2 / count - mean ** 2).clamp(min=1e-24).sqrt()
    return mean.float(), std.float()
