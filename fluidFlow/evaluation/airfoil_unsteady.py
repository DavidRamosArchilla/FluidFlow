"""Dataset-specific evaluators for the unsteady airfoil case.

Ports ``scripts/evaluate_airfoil_unsteady.py`` (field metrics, RMSE-vs-frame,
Mach-AoA scatter, prediction / error GIFs) and
``scripts/fft_airfoil_unsteady.py`` (temporal spectra, probe traces) into
reusable :class:`Evaluator` classes with the same structure and outputs.

All evaluators expect denormalized (physical units) fields with shape
``(N, F, C, L)``: N simulations, F frames, C channels, L mesh nodes.
"""

import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .base import Evaluator
from .regression import RegressionEvaluator, _to_numpy


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def shock_scores(gt):
    """Pressure-range score per sample, shape (N,). Expects (N, F, C, L) with p at channel 2."""
    p = np.asarray(gt)[:, :, 2, :]
    return p.max(axis=(1, 2)) - p.min(axis=(1, 2))


def filter_pool(n_samples, conds, mach_min=None, aoa_lim=None):
    """Keep sample indices satisfying the Mach/AoA filters. conds is (N, 2) Mach/alpha."""
    pool = np.arange(n_samples)
    if mach_min is not None:
        pool = pool[conds[pool, 0] >= mach_min]
    if aoa_lim is not None:
        pool = pool[np.abs(conds[pool, 1]) <= aoa_lim]
    if len(pool) == 0:
        raise RuntimeError("no test samples satisfy the Mach/AoA filter")
    return pool


def select_samples(gt, conds, samples="auto", auto_by="shock",
                   mach_min=None, aoa_lim=None, norm_stats=None, n_top=3):
    """Select sample indices: 'auto' (top-n_top), 'auto:K' or a comma list.

    auto_by: "shock" (largest pressure range) or "movement" (strongest
    temporal fluctuation in normalized space, needs norm_stats).
    """
    gt = np.asarray(gt)
    conds = np.asarray(conds)
    n = gt.shape[0]
    pool = filter_pool(n, conds, mach_min, aoa_lim)
    if isinstance(samples, str) and samples == "auto":
        k = n_top
    elif isinstance(samples, str) and samples.startswith("auto:"):
        k = int(samples.split(":")[1])
    else:
        if isinstance(samples, (int, np.integer)):
            samples = [samples]
        elif isinstance(samples, str):
            samples = [int(x) for x in samples.split(",")]
        sel = [int(s) for s in samples]
        assert all(0 <= s < n for s in sel), f"sample out of range (N={n})"
        return sel
    if auto_by == "movement":
        if norm_stats is None:
            raise ValueError("auto_by='movement' needs norm_stats for normalization")
        fmean = np.asarray(norm_stats["fields_mean"]).reshape(1, 1, -1, 1)
        fstd = np.asarray(norm_stats["fields_std"]).reshape(1, 1, -1, 1)
        gn = (gt[pool] - fmean) / fstd
        move = gn.std(axis=1).mean(axis=(1, 2))
        order = pool[np.argsort(-move)]
    else:
        scores = shock_scores(gt)
        order = sorted(pool.tolist(), key=lambda i: -scores[i])
        order = np.asarray(order, dtype=int)
    return [int(i) for i in order[:k]]


def mean_power_spectrum(sig):
    """Mean (node-averaged) one-sided power spectrum of sig (F, L)."""
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig - sig.mean(axis=0, keepdims=True)  # drop DC for spectral shape
    X = np.fft.rfft(sig, axis=0)
    return (np.abs(X) ** 2).mean(axis=1)  # (F//2 + 1,)


def spectral_metrics(p_pred, p_gt):
    p_pred = np.asarray(p_pred, dtype=np.float64)
    p_gt = np.asarray(p_gt, dtype=np.float64)
    denom = float(np.linalg.norm(p_gt)) + 1e-30
    rel_l2 = float(np.linalg.norm(p_pred - p_gt) / denom)
    # log distance only where GT carries significant energy (60 dB range)
    m = p_gt > 1e-6 * float(p_gt.max())
    if m.sum() == 0:
        m = np.ones_like(p_gt, dtype=bool)
    eps = 1e-30
    log_mae = float(np.mean(np.abs(np.log(p_pred[m] + eps) - np.log(p_gt[m] + eps))))
    return {"spectral_rel_l2": rel_l2, "log_spectral_mae": log_mae}


def mean_welch_spectrum(sig, fs, nperseg):
    """Mean (node-averaged) Welch PSD of sig (F, L)."""
    from scipy.signal import welch

    sig = np.asarray(sig, dtype=np.float64)
    sig = sig - sig.mean(axis=0, keepdims=True)
    nperseg = min(int(nperseg), sig.shape[0])
    f, Pxx = welch(sig, fs=fs, nperseg=nperseg, axis=0, average="mean")
    return f, Pxx.mean(axis=1)


def print_metrics_block(title, metrics):
    print(f"\n{title}")
    for key, value in metrics.items():
        if key == "mre":
            print(f"  {key}: {value:.4f}%")
        elif key == "r2":
            print(f"  {key}: {value:.4f}")
        else:
            if value < 1e-3 or value > 1e3:
                print(f"  {key}: {value:.4e}")
            else:
                print(f"  {key}: {value:.4f}")


# --------------------------------------------------------------------------
# Field evaluator (metrics + RMSE curves + GIFs)
# --------------------------------------------------------------------------

class AirfoilUnsteadyEvaluator(Evaluator):
    """Global + per-channel regression metrics, RMSE-vs-frame, Mach-AoA
    scatter and prediction / absolute-error GIFs for unsteady airfoil fields.

    Args:
        out_dir: folder where metrics.json, RMSE plots and GIFs are written.
        channel_names: names per channel, e.g. ("u", "v", "p").
        space: "physical", "normalized" (needs norm_stats) or "both".
        sample_idx: GIF sample, "auto", "auto:K" or int.
        auto_by: "shock" or "movement".
        mach_min / aoa_lim: filters for the auto-selection pool.
        frame_stride / fps / dpi / levels / cmap_pred / cmap_err / zoom:
            GIF and plot rendering options.
    """

    def __init__(
        self,
        out_dir,
        channel_names=("u", "v", "p"),
        space="physical",
        sample_idx="auto",
        auto_by="shock",
        mach_min=None,
        aoa_lim=None,
        frame_stride=6,
        fps=10,
        dpi=100,
        levels=80,
        cmap_pred="viridis",
        cmap_err="YlOrRd",
        zoom=None,
    ):
        self.out_dir = out_dir
        self.channel_names = list(channel_names)
        self.space = space
        self.sample_idx = sample_idx
        self.auto_by = auto_by
        self.mach_min = mach_min
        self.aoa_lim = aoa_lim
        self.frame_stride = frame_stride
        self.fps = fps
        self.dpi = dpi
        self.levels = levels
        self.cmap_pred = cmap_pred
        self.cmap_err = cmap_err
        self.zoom = zoom
        self.regression = RegressionEvaluator()
        self._metrics = None

    def __call__(self, y_true, y_pred, mesh_pos=None, node_type=None,
                 cells=None, conds=None, norm_stats=None):
        """Evaluate predictions against ground truth.

        Args:
            y_true / y_pred: (N, F, C, L) fields in physical units.
            mesh_pos: (L, 2) node coordinates (needed for GIFs).
            node_type: (L,) node types, 2 = wall (needed for GIF auto-zoom).
            cells: (M, 3) mesh triangles (needed for GIFs).
            conds: (N, 2) physical Mach/alpha per sample (needed for
                selection labels and the Mach-AoA scatter).
            norm_stats: dict with fields_mean/fields_std (needed for
                space="normalized"/"both" and auto_by="movement").

        Returns:
            dict with global and per-channel metrics per space.
        """
        y_true = _to_numpy(y_true).astype(np.float64)
        y_pred = _to_numpy(y_pred).astype(np.float64)
        assert y_true.shape == y_pred.shape, f"GT {y_true.shape} != pred {y_pred.shape}"
        n, n_frames, n_channels, n_nodes = y_true.shape
        assert n_channels == len(self.channel_names)
        conds = _to_numpy(conds) if conds is not None else None

        os.makedirs(self.out_dir, exist_ok=True)

        spaces = {"physical": (y_pred, y_true)}
        if self.space in ("normalized", "both"):
            if norm_stats is None:
                raise ValueError('space="normalized"/"both" needs norm_stats')
            fmean = np.asarray(norm_stats["fields_mean"]).reshape(1, 1, -1, 1)
            fstd = np.asarray(norm_stats["fields_std"]).reshape(1, 1, -1, 1)
            spaces["normalized"] = ((y_pred - fmean) / fstd, (y_true - fmean) / fstd)
        if self.space == "normalized":
            del spaces["physical"]

        all_metrics = {}
        for space_name, (a, b) in spaces.items():
            suffix = "" if space_name == "physical" else "_normalized"
            print(f"\n=== Global {space_name} metrics ===")
            global_metrics = self.regression(b, a)
            print_metrics_block(
                f"Global {space_name} (all channels, samples, frames, nodes):",
                global_metrics)

            print(f"\n=== Per-channel {space_name} metrics ===")
            per_channel = {}
            for c, name in enumerate(self.channel_names):
                m = self.regression(b[:, :, c], a[:, :, c])
                per_channel[name] = {k: float(v) for k, v in m.items()}
                print_metrics_block(f"Channel {name}:", per_channel[name])

            with open(os.path.join(self.out_dir, f"metrics{suffix}.json"), "w") as f:
                json.dump({"space": space_name, "global": {k: float(v) for k, v in global_metrics.items()},
                           "per_channel": per_channel, "channels": self.channel_names,
                           "pred_shape": list(a.shape)}, f, indent=2)
            print(f"\nSaved {self.out_dir}/metrics{suffix}.json")

            self._rmse_vs_frame(a, b, space_name, suffix, n_frames, n_channels)
            if conds is not None:
                self._mach_aoa_scatter(a, b, space_name, suffix, conds, n, n_channels)
            all_metrics[space_name] = {"global": {k: float(v) for k, v in global_metrics.items()},
                                       "per_channel": per_channel}

        if mesh_pos is not None and cells is not None and conds is not None:
            self._gifs(y_true, y_pred, mesh_pos, node_type, cells, conds,
                       n_frames, n_channels, n_nodes, norm_stats)
        else:
            print("\nSkipping GIFs (mesh_pos, cells and conds are all required).")

        self._metrics = all_metrics
        print(f"\nDone. All outputs in {self.out_dir}/")
        return all_metrics

    # -- sub-steps ------------------------------------------------------

    def _rmse_vs_frame(self, a, b, space_name, suffix, n_frames, n_channels):
        print(f"\nComputing per-frame {space_name} RMSE ...")
        rmse_global = np.zeros(n_frames)
        rmse_chan = np.zeros((n_channels, n_frames))
        for t in range(n_frames):
            d = a[:, t] - b[:, t]  # (N, C, L)
            rmse_global[t] = float(np.sqrt(np.mean(d ** 2)))
            for c in range(n_channels):
                rmse_chan[c, t] = float(np.sqrt(np.mean(d[:, c] ** 2)))
            if (t + 1) % 100 == 0:
                print(f"  frame {t + 1}/{n_frames}")
        print(f"\nComputing per-sample {space_name} RMSE ...")
        n = a.shape[0]
        rmse_sample = np.zeros((n, n_channels))
        for i in range(n):
            d = a[i] - b[i]  # (F, C, L)
            for c in range(n_channels):
                rmse_sample[i, c] = float(np.sqrt(np.mean(d[:, c] ** 2)))
            if (i + 1) % 20 == 0:
                print(f"  sample {i + 1}/{n}")
        rmse_sample_global = np.sqrt((rmse_sample ** 2).mean(axis=1))
        np.savez(os.path.join(self.out_dir, f"rmse_vs_frame{suffix}.npz"),
                 rmse_global=rmse_global, rmse_chan=rmse_chan,
                 rmse_sample_chan=rmse_sample,
                 rmse_sample_global=rmse_sample_global,
                 channels=np.array(self.channel_names))
        frames = np.arange(n_frames)

        fig, ax1 = plt.subplots(figsize=(10, 5))
        # velocity channels on the left axis, p/rho (+ global) on the right:
        # pressure RMSE lives on a much larger scale than u/v (physical space).
        left_idx = [c for c in range(n_channels) if self.channel_names[c] in ("u", "v")]
        right_idx = [c for c in range(n_channels) if c not in left_idx]
        for c in left_idx:
            ax1.plot(frames, rmse_chan[c], label=f"{self.channel_names[c]}", linewidth=1.5)
        ax1.set_xlabel("frame")
        ax1.set_ylabel(f"RMSE {space_name} (u, v)")
        ax1.grid(alpha=0.3)
        ax2 = ax1.twinx()
        for c in right_idx:
            ax2.plot(frames, rmse_chan[c], label=f"{self.channel_names[c]}",
                     linewidth=1.5, linestyle="--")
        ax2.plot(frames, rmse_global, label="global", linewidth=2, color="black")
        ax2.set_ylabel(f"RMSE {space_name} ({', '.join([self.channel_names[c] for c in right_idx])}, global)")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best")
        plt.title(f"Per-frame {space_name} RMSE on test set (sample-averaged, N={n})")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, f"rmse_vs_frame{suffix}.png"), dpi=self.dpi)
        plt.close()
        print(f"Saved {self.out_dir}/rmse_vs_frame{suffix}.png")

    def _mach_aoa_scatter(self, a, b, space_name, suffix, conds, n, n_channels):
        rmse_sample = np.zeros((n, n_channels))
        for i in range(n):
            d = a[i] - b[i]
            for c in range(n_channels):
                rmse_sample[i, c] = float(np.sqrt(np.mean(d[:, c] ** 2)))
        rmse_sample_global = np.sqrt((rmse_sample ** 2).mean(axis=1))
        mach_all, alpha_all = conds[:, 0], conds[:, 1]
        worst = np.argsort(-rmse_sample_global)[:3]
        print(f"  worst samples ({space_name} global RMSE): " +
              ", ".join(f"{k} (rmse {rmse_sample_global[k]:.4g}, "
                        f"Mach {mach_all[k]:.3f}, AoA {alpha_all[k]:.2f}°)"
                        for k in worst))
        panels = [("global", rmse_sample_global)] + [
            (self.channel_names[c], rmse_sample[:, c]) for c in range(n_channels)]
        ncols = 2
        nrows = int(np.ceil(len(panels) / ncols))
        fig3, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                                 squeeze=False)
        for ax, (label, vals) in zip(axs.ravel(), panels):
            sc = ax.scatter(mach_all, alpha_all, c=vals, cmap="plasma")
            fig3.colorbar(sc, ax=ax, label=f"RMSE {space_name}")
            ax.set_xlabel("Mach")
            ax.set_ylabel("AoA [deg]")
            ax.set_title(f"{label}")
            ax.grid(alpha=0.3)
        for ax in axs.ravel()[len(panels):]:
            ax.axis("off")
        fig3.suptitle(f"Per-simulation RMSE over the test set ({space_name}, N={n})")
        fig3.tight_layout()
        mach_aoa_png = f"mach_aoa_rmse{suffix}.png"
        fig3.savefig(os.path.join(self.out_dir, mach_aoa_png), dpi=self.dpi)
        plt.close(fig3)
        print(f"Saved {self.out_dir}/{mach_aoa_png}")

    def _gifs(self, gt, pred, mesh_pos, node_type, cells, conds,
              n_frames, n_channels, n_nodes, norm_stats):
        mesh_pos = np.asarray(mesh_pos)
        cells = np.asarray(cells)
        sel = select_samples(gt, conds, samples=self.sample_idx,
                             auto_by=self.auto_by, mach_min=self.mach_min,
                             aoa_lim=self.aoa_lim, norm_stats=norm_stats)
        s = sel[0]
        print(f"\nBuilding GIFs for sample {s} ...")
        np.save(os.path.join(self.out_dir, "mesh_pos.npy"), mesh_pos)

        mx, my = mesh_pos[:, 0], mesh_pos[:, 1]
        if self.zoom is not None:
            xmin, xmax, ymin, ymax = self.zoom
        else:
            # zoom around the airfoil wall nodes (node_type == 2) + margin
            wall = mx[node_type == 2], my[node_type == 2]
            chord = float(wall[0].max() - wall[0].min())
            m = 0.65 * chord
            xmin, xmax = float(wall[0].min() - m), float(wall[0].max() + m)
            yc = float(0.5 * (wall[1].min() + wall[1].max()))
            half_y = (xmax - xmin) / 2 / 1.5
            ymin, ymax = yc - half_y, yc + half_y
        print(f"  zoom x [{xmin:.3f}, {xmax:.3f}] y [{ymin:.3f}, {ymax:.3f}]")

        gt_s = gt[s]    # (F, C, L)
        pred_s = pred[s]
        mach_s, alpha_s = float(conds[s, 0]), float(conds[s, 1])
        tag = f"Mach {mach_s:.3f}, AoA {alpha_s:.2f}°"
        frame_idx = list(range(0, n_frames, self.frame_stride))
        print(f"  {len(frame_idx)} frames (stride {self.frame_stride})")

        clim = []
        for c in range(n_channels):
            lo = float(min(gt_s[frame_idx, c].min(), pred_s[frame_idx, c].min()))
            hi = float(max(gt_s[frame_idx, c].max(), pred_s[frame_idx, c].max()))
            clim.append((lo, hi))
        err_max = [float(np.abs(gt_s[frame_idx, c] - pred_s[frame_idx, c]).max())
                   for c in range(n_channels)]

        def build_gif(mode, filename, title, cmap):
            from matplotlib.tri import Triangulation

            tri = Triangulation(mx, my, cells)
            fig, axes = plt.subplots(1, n_channels, figsize=(5 * n_channels, 3.4))
            if n_channels == 1:
                axes = [axes]
            levels = []
            for c in range(n_channels):
                if mode == "pred":
                    vmin, vmax = clim[c]
                else:
                    vmin, vmax = 0.0, err_max[c]
                levels.append(np.linspace(vmin, vmax, self.levels))

            def frame_values(t, c):
                if mode == "pred":
                    return pred_s[t, c]
                return np.abs(gt_s[t, c] - pred_s[t, c])

            def draw_frame(k):
                t = frame_idx[k]
                for c, ax in enumerate(axes):
                    ax.clear()
                    ax.tricontourf(tri, frame_values(t, c), levels=levels[c],
                                   cmap=cmap, extend="both")
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)
                    ax.set_aspect("equal")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f"pred {self.channel_names[c]}" if mode == "pred"
                                 else f"|error| {self.channel_names[c]}")
                sup.set_text(f"{title} — {tag}, frame {t}")
                return sup,

            sup = fig.suptitle(f"{title} — {tag}, frame {frame_idx[0]}")
            draw_frame(0)
            for ax in axes:
                fig.colorbar(ax.collections[0], ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()

            anim = FuncAnimation(fig, draw_frame, frames=len(frame_idx), blit=False)
            anim.save(os.path.join(self.out_dir, filename), writer="pillow",
                      fps=self.fps, dpi=self.dpi)
            plt.close(fig)
            print(f"Saved {self.out_dir}/{filename}")

        build_gif("pred", "predictions.gif", "Prediction", cmap=self.cmap_pred)
        build_gif("err", "abs_error.gif", "Absolute error", cmap=self.cmap_err)

    def print_metrics(self):
        if self._metrics is None:
            raise ValueError("No metrics have been calculated yet.")
        for space_name, space_metrics in self._metrics.items():
            print_metrics_block(f"Global {space_name}:", space_metrics["global"])
            for name, m in space_metrics["per_channel"].items():
                print_metrics_block(f"Channel {name} ({space_name}):", m)


# --------------------------------------------------------------------------
# FFT evaluator (temporal spectra + probe traces)
# --------------------------------------------------------------------------

class AirfoilUnsteadyFFTEvaluator(Evaluator):
    """Temporal frequency analysis of predictions vs GT, per channel:
    mean power spectra, dominant frequencies, spectral error metrics,
    Welch PSDs and probe time traces.

    Args:
        out_dir: folder where spectra plots, traces, spectra.npz and
            fft_metrics.json are written.
        dt: timestep in seconds (sampling rate fs = 1/dt).
        channel_names: names per channel, e.g. ("u", "v", "p").
        samples: "auto" (top-3 shock-score), "auto:K" or comma list / list.
        mach_min / aoa_lim: filters for the auto-selection pool.
        nperseg: Welch segment length (freq. resolution = fs/nperseg).
        dpi: figure resolution.
    """

    def __init__(
        self,
        out_dir,
        dt=0.0002,
        channel_names=("u", "v", "p"),
        samples="auto",
        mach_min=None,
        aoa_lim=None,
        nperseg=300,
        dpi=100,
    ):
        self.out_dir = out_dir
        self.dt = dt
        self.channel_names = list(channel_names)
        self.samples = samples
        self.mach_min = mach_min
        self.aoa_lim = aoa_lim
        self.nperseg = nperseg
        self.dpi = dpi
        self._metrics = None

    def __call__(self, y_true, y_pred, conds=None, mesh_pos=None, norm_stats=None):
        """Run the spectral analysis.

        Args:
            y_true / y_pred: (N, F, C, L) fields in physical units.
            conds: (N, 2) physical Mach/alpha per sample (for labels).
            mesh_pos: (L, 2) node coordinates (for probe location printout).
            norm_stats: optional, forwarded to sample selection
                (needed for samples="auto:movement"-style criteria).

        Returns:
            dict with dt/fs, selected samples and per-sample metrics.
        """
        gt = _to_numpy(y_true).astype(np.float64)
        pred = _to_numpy(y_pred).astype(np.float64)
        assert gt.shape == pred.shape
        n, n_frames, n_channels, n_nodes = gt.shape
        assert n_channels == len(self.channel_names)
        conds = _to_numpy(conds) if conds is not None else np.zeros((n, 2))
        mesh_pos = np.asarray(mesh_pos) if mesh_pos is not None else None

        os.makedirs(self.out_dir, exist_ok=True)
        fs = 1.0 / self.dt
        print(f"dt={self.dt} s, sampling {fs:.1f} Hz, Nyquist {fs / 2:.1f} Hz")
        freqs = np.fft.rfftfreq(n_frames, self.dt)

        sel = select_samples(gt, conds, samples=self.samples,
                             mach_min=self.mach_min, aoa_lim=self.aoa_lim,
                             norm_stats=norm_stats)
        print(f"\nSelected samples {sel}:")
        for s in sel:
            print(f"  sample {s}: Mach {conds[s, 0]:.3f}, alpha {conds[s, 1]:.2f} deg")

        all_metrics = {}
        spectra = {"freqs": freqs, "channels": np.array(self.channel_names)}
        for s in sel:
            tag = f"Mach {conds[s, 0]:.3f}, AoA {conds[s, 1]:.2f}°"
            sm = {"mach": float(conds[s, 0]), "alpha": float(conds[s, 1])}
            # probe = node with largest GT pressure fluctuation
            probe = int(gt[s, :, 2, :].std(axis=0).argmax())
            sm["probe_node"] = probe
            if mesh_pos is not None:
                print(f"\nsample {s} ({tag}), probe node {probe} "
                      f"at ({mesh_pos[probe, 0]:.3f}, {mesh_pos[probe, 1]:.3f})")
            else:
                print(f"\nsample {s} ({tag}), probe node {probe}")

            fig, axes = plt.subplots(1, n_channels, figsize=(5 * n_channels, 3.6))
            if n_channels == 1:
                axes = [axes]
            for c, (ax, name) in enumerate(zip(axes, self.channel_names)):
                psd_gt = mean_power_spectrum(gt[s, :, c, :])
                psd_pred = mean_power_spectrum(pred[s, :, c, :])
                spectra[f"sample{s}_{name}_gt"] = psd_gt
                spectra[f"sample{s}_{name}_pred"] = psd_pred
                ax.loglog(freqs[1:], psd_gt[1:], color="black", lw=1.5, label="GT")
                ax.loglog(freqs[1:], psd_pred[1:], color="tab:red", lw=1.2, label="pred")
                ax.set_xlabel("frequency [Hz]")
                ax.set_ylabel("mean power")
                ax.set_title(f"{name}")
                ax.grid(alpha=0.3, which="both")
                i_gt = 1 + int(np.argmax(psd_gt[1:]))
                i_pr = 1 + int(np.argmax(psd_pred[1:]))
                f_gt, f_pr = float(freqs[i_gt]), float(freqs[i_pr])
                ax.axvline(f_gt, color="black", ls=":", lw=1)
                ax.axvline(f_pr, color="tab:red", ls=":", lw=1)
                m = spectral_metrics(psd_pred[1:], psd_gt[1:])
                m.update({"f_dom_gt": f_gt, "f_dom_pred": f_pr,
                          "f_dom_rel_err": abs(f_pr - f_gt) / (abs(f_gt) + 1e-30)})
                sm[name] = {k: float(v) for k, v in m.items()}
                print(f"  {name}: f_dom GT {f_gt:.2f} Hz vs pred {f_pr:.2f} Hz | "
                      f"spectral relL2 {m['spectral_rel_l2']:.4f} | "
                      f"log-spec MAE {m['log_spectral_mae']:.4f}")
            axes[0].legend(loc="best", fontsize=9)
            fig.suptitle(f"Mean power spectra — sample {s} ({tag})")
            fig.tight_layout()
            fig.savefig(os.path.join(self.out_dir, f"psd_sample{s}.png"), dpi=self.dpi)
            plt.close(fig)
            print(f"  saved psd_sample{s}.png")

            # Welch PSD (smoother curves): averaged over segments AND nodes
            fw, _ = mean_welch_spectrum(gt[s, :, 0, :], fs, self.nperseg)
            print(f"  Welch: nperseg={min(self.nperseg, n_frames)} -> "
                  f"{fs / min(self.nperseg, n_frames):.1f} Hz resolution, {len(fw)} bins")
            spectra["welch_freqs"] = fw
            figw, axesw = plt.subplots(1, n_channels, figsize=(5 * n_channels, 3.6))
            if n_channels == 1:
                axesw = [axesw]
            for c, (ax, name) in enumerate(zip(axesw, self.channel_names)):
                _, w_gt = mean_welch_spectrum(gt[s, :, c, :], fs, self.nperseg)
                _, w_pred = mean_welch_spectrum(pred[s, :, c, :], fs, self.nperseg)
                spectra[f"sample{s}_{name}_welch_gt"] = w_gt
                spectra[f"sample{s}_{name}_welch_pred"] = w_pred
                ax.loglog(fw[1:], w_gt[1:], color="black", lw=1.5, label="GT")
                ax.loglog(fw[1:], w_pred[1:], color="tab:red", lw=1.2, label="pred")
                ax.set_xlabel("frequency [Hz]")
                ax.set_ylabel("mean Welch power")
                ax.set_title(f"{name}")
                ax.grid(alpha=0.3, which="both")
                i_gt = 1 + int(np.argmax(w_gt[1:]))
                i_pr = 1 + int(np.argmax(w_pred[1:]))
                f_gt, f_pr = float(fw[i_gt]), float(fw[i_pr])
                ax.axvline(f_gt, color="black", ls=":", lw=1)
                ax.axvline(f_pr, color="tab:red", ls=":", lw=1)
                mw = spectral_metrics(w_pred[1:], w_gt[1:])
                sm[name].update({f"welch_{k}": float(v) for k, v in mw.items()})
                sm[name].update({"welch_f_dom_gt": f_gt, "welch_f_dom_pred": f_pr})
                print(f"  {name} welch: f_dom GT {f_gt:.2f} Hz vs pred {f_pr:.2f} Hz | "
                      f"relL2 {mw['spectral_rel_l2']:.4f} | "
                      f"log-spec MAE {mw['log_spectral_mae']:.4f}")
            axesw[0].legend(loc="best", fontsize=9)
            figw.suptitle(f"Mean Welch spectra — sample {s} ({tag})")
            figw.tight_layout()
            figw.savefig(os.path.join(self.out_dir, f"welch_sample{s}.png"), dpi=self.dpi)
            plt.close(figw)
            print(f"  saved welch_sample{s}.png")

            # probe time traces
            t = np.arange(n_frames) * self.dt
            fig2, axes2 = plt.subplots(n_channels, 1, figsize=(9, 2.4 * n_channels),
                                       sharex=True)
            if n_channels == 1:
                axes2 = [axes2]
            for c, (ax, name) in enumerate(zip(axes2, self.channel_names)):
                ax.plot(t, gt[s, :, c, probe], color="black", lw=1, label="GT")
                ax.plot(t, pred[s, :, c, probe], color="tab:red", lw=1, label="pred")
                ax.set_ylabel(name)
                ax.grid(alpha=0.3)
            axes2[0].legend(loc="best", fontsize=9)
            axes2[-1].set_xlabel("time [s]")
            fig2.suptitle(f"Probe traces at node {probe} — sample {s} ({tag})")
            fig2.tight_layout()
            fig2.savefig(os.path.join(self.out_dir, f"traces_sample{s}.png"), dpi=self.dpi)
            plt.close(fig2)
            print(f"  saved traces_sample{s}.png")
            all_metrics[f"sample{s}"] = sm

        np.savez(os.path.join(self.out_dir, "spectra.npz"), **spectra)
        with open(os.path.join(self.out_dir, "fft_metrics.json"), "w") as f:
            json.dump({"dt": self.dt, "fs": fs, "samples": sel,
                       "channels": self.channel_names, "metrics": all_metrics}, f, indent=2)
        print(f"\nDone. All outputs in {self.out_dir}/")
        self._metrics = all_metrics
        return {"dt": self.dt, "fs": fs, "samples": sel,
                "channels": self.channel_names, "metrics": all_metrics}

    def print_metrics(self):
        if self._metrics is None:
            raise ValueError("No metrics have been calculated yet.")
        for sample, sm in self._metrics.items():
            print(f"\n{sample} (Mach {sm['mach']:.3f}, AoA {sm['alpha']:.2f}°):")
            for ch in self.channel_names:
                if ch in sm:
                    print_metrics_block(f"  {ch}:", sm[ch])
