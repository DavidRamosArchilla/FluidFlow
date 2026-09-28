import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from accelerate import Accelerator, DataLoaderConfiguration
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm.auto import tqdm

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt


def exists(val):
    return val is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

class TrainerVAE:
    """Accelerate-based trainer for variational autoencoders.

    Model-agnostic: works with any VAE exposing ``cfg.kl_weight``,
    ``(x_hat, posterior) = model(x)`` and ``model.loss(x, x_hat, posterior)``
    (e.g. :class:`fluidFlow.video_vae.AutoencoderKLLTX`).
    """
    def __init__(
        self,
        model: nn.Module,
        dataset,
        *,
        # ── data ──────────────────────────────────────────────────
        train_batch_size: int = 16,
        dataset_test=None,
        # ── optimisation ──────────────────────────────────────────
        train_lr: float = 1e-4,
        adam_betas: tuple = (0.9, 0.999),
        adam_weight_decay: float = 1e-2,
        train_num_steps: int = 100_000,
        gradient_accumulate_every: int = 1,
        max_grad_norm: Optional[float] = 1.0,
        # ── loss ──────────────────────────────────────────────────
        kl_weight: Optional[float] = None,
        # ── lr scheduler ──────────────────────────────────────────
        eta_min_scheduler: Optional[float] = None,
        # ── mixed precision / accelerate ──────────────────────────
        amp: bool = False,
        mixed_precision_type: str = "bf16",
        split_batches: bool = True,
        use_cpu: bool = False,
        # ── checkpointing ─────────────────────────────────────────
        results_folder: str = "./results_vae",
        save_every: int = 1000,
        # ── misc ──────────────────────────────────────────────────
        compile_model: bool = False,
        num_workers: int = 4,
    ):
        super().__init__()

        # ── accelerator ───────────────────────────────────────────────────────
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision_type if amp else "no",
            cpu=use_cpu,
            dataloader_config=DataLoaderConfiguration(split_batches=split_batches),
            gradient_accumulation_steps=gradient_accumulate_every,
        )

        # ── model ─────────────────────────────────────────────────────────────
        self.model = model

        # kl_weight can be overridden here without changing the config
        if exists(kl_weight):
            self.model.cfg.kl_weight = kl_weight

        # ── training state ────────────────────────────────────────────────────
        self.step = 0
        self.train_num_steps = train_num_steps
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm
        self.save_every = save_every

        # ── data ──────────────────────────────────────────────────────────────
        dl = DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

        if exists(dataset_test):
            self.dl_test = DataLoader(
                dataset_test,
                batch_size=train_batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=num_workers,
            )
        else:
            self.dl_test = None

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # ── optimiser ─────────────────────────────────────────────────────────
        # eps follows mixed-precision convention (same as the original Trainer1D)
        eps = 1e-6 if amp and mixed_precision_type in ("fp16", "bf16") else 1e-8
        self.opt = AdamW(
            model.parameters(),
            lr=train_lr,
            betas=adam_betas,
            weight_decay=adam_weight_decay,
            fused=not use_cpu,   # fused kernel only available on CUDA
            eps=eps,
        )

        # ── lr scheduler ──────────────────────────────────────────────────────
        self.use_lr_scheduler = exists(eta_min_scheduler)
        if self.use_lr_scheduler:
            warmup_steps = 1000
            warmup = LinearLR(self.opt, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
            cosine = CosineAnnealingLR(self.opt, T_max=train_num_steps - warmup_steps, eta_min=eta_min_scheduler)
            self.scheduler = SequentialLR(self.opt, schedulers=[warmup, cosine], milestones=[warmup_steps])
            self.scheduler = self.accelerator.prepare_scheduler(self.scheduler)

        # ── results folder ────────────────────────────────────────────────────
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        # ── prepare everything with accelerate ────────────────────────────────
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        if exists(self.dl_test):
            self.dl_test = self.accelerator.prepare(self.dl_test)

        if compile_model:
            self.accelerator.print("Compiling model …")
            self.model = torch.compile(self.model)
            self.accelerator.print("Model compiled.")

        # ── loss history ──────────────────────────────────────────────────────
        self.loss_history: List[dict] = []      # {"step", "rec", "kl", "total"}
        self.test_loss_history: List[dict] = [] # {"step", "rec", "kl", "total"}

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def _unwrapped_model(self) -> nn.Module:
        return self.accelerator.unwrap_model(self.model)

    # ── checkpointing ─────────────────────────────────────────────────────────

    def save(self, step: Optional[int] = None, tag: Optional[str] = None):
        """
        Save model weights + optimiser + scheduler + loss history.

        Filenames:
            vae_{step:08d}.pt   if step is given
            vae_{tag}.pt        if a custom tag is given (e.g. 'best')
            vae_latest.pt       always updated so you can resume easily
        """
        if not self.is_main:
            return
        lr = self.opt.param_groups[0]['lr']
        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "loss_history": self.loss_history,
            "test_loss_history": self.test_loss_history,
            "lr": lr,
        }
        if self.use_lr_scheduler:
            data["scheduler"] = self.scheduler.state_dict()

        # always write a "latest" file for easy resuming
        torch.save(data, self.results_folder / "vae_latest.pt")

        if exists(step):
            torch.save(data, self.results_folder / f"vae_{step:08d}.pt")
        if exists(tag):
            torch.save(data, self.results_folder / f"vae_{tag}.pt")

    def load(self, path: Optional[str] = None):
        """
        Load a checkpoint.  If *path* is None, looks for vae_latest.pt in
        results_folder.
        """
        path = Path(path) if exists(path) else self.results_folder / "vae_latest.pt"

        if not path.exists():
            raise FileNotFoundError(f"No checkpoint found at {path}")

        data = torch.load(path, map_location=self.device)

        unwrapped = self._unwrapped_model()
        unwrapped.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])

        if self.use_lr_scheduler and "scheduler" in data:
            self.scheduler.load_state_dict(data["scheduler"])

        self.loss_history      = data.get("loss_history", [])
        self.test_loss_history = data.get("test_loss_history", [])

        if "lr" in data:
            print(f"Setting loaded learning rate to {data['lr']}")
            for param_group in self.opt.param_groups:
                param_group['lr'] = data['lr']

        self.accelerator.print(f"Loaded checkpoint from {path}  (step {self.step})")

    # ── plotting ──────────────────────────────────────────────────────────────

    def _save_loss_plot(self):
        """Save a PNG with train (and optional test) rec / kl / total losses."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"VAE losses — step {self.step}", fontsize=12)

        metrics = ("rec", "kl", "total")
        titles  = ("Reconstruction loss", "KL loss", "Total loss")

        for ax, key, title in zip(axes, metrics, titles):
            if self.loss_history:
                steps_tr = [e["step"] for e in self.loss_history]
                vals_tr  = [e[key]   for e in self.loss_history]
                ax.plot(steps_tr, vals_tr, linewidth=0.8, alpha=0.6, label="train")

                # smoothed train curve (simple moving average)
                if len(vals_tr) >= 50:
                    window = max(1, len(vals_tr) // 50)
                    smoothed = [
                        sum(vals_tr[max(0, i - window): i + 1]) /
                        len(vals_tr[max(0, i - window): i + 1])
                        for i in range(len(vals_tr))
                    ]
                    ax.plot(steps_tr, smoothed, linewidth=1.5, label="train (smooth)")

            if self.test_loss_history:
                steps_te = [e["step"] for e in self.test_loss_history]
                vals_te  = [e[key]   for e in self.test_loss_history]
                ax.plot(steps_te, vals_te, "o-", linewidth=1.5,
                        markersize=4, label="test")

            ax.set_title(title)
            ax.set_xlabel("step")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")

        plt.tight_layout()
        latest_path = self.results_folder / "losses_latest.png"
        plt.savefig(latest_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    # ── validation helper ─────────────────────────────────────────────────────

    @torch.no_grad()
    def _eval_test(self) -> Optional[dict]:
        if not exists(self.dl_test):
            return None

        self.model.eval()
        total_rec = total_kl = total_loss = 0.0
        n = 0

        for batch in self.dl_test:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x_hat, posterior = self.model(x)
            loss, rec, kl = self._unwrapped_model().loss(x, x_hat, posterior)

            # gather across processes
            rec_g, kl_g, loss_g = self.accelerator.gather_for_metrics(
                (rec.unsqueeze(0), kl.unsqueeze(0), loss.unsqueeze(0))
            )
            total_rec  += rec_g.mean().item()
            total_kl   += kl_g.mean().item()
            total_loss += loss_g.mean().item()
            n += 1

        self.model.train()
        return {"rec": total_rec / n, "kl": total_kl / n, "total": total_loss / n}

    # ── main training loop ────────────────────────────────────────────────────

    def train(self):
        accelerator = self.accelerator
        model = self.model
        cfg = self._unwrapped_model().cfg

        accelerator.print(
            f"\nStarting training: steps={self.train_num_steps}  "
            f"batch={self.batch_size}  "
            f"grad_accum={self.gradient_accumulate_every}  "
            f"kl_weight={cfg.kl_weight:.2e}\n"
        )

        best_test_loss = float("inf")

        # tqdm only on the main process so it doesn't duplicate across GPUs
        pbar = tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not self.is_main,
            dynamic_ncols=True,
            desc="Training VAE",
        )

        model.train()
        while self.step < self.train_num_steps:
            # ── gradient accumulation loop ────────────────────────────────────
            acc_rec = acc_kl = acc_total = 0.0

            for _ in range(self.gradient_accumulate_every):
                batch = next(self.dl)
                x = batch[0] if isinstance(batch, (list, tuple)) else batch

                with accelerator.accumulate(model):
                    x_hat, posterior = model(x)
                    loss, rec, kl = self._unwrapped_model().loss(x, x_hat, posterior)
                    accelerator.backward(loss)

                acc_rec   += rec.item()
                acc_kl    += kl.item()
                acc_total += loss.item()

            # average over accumulation steps
            acc_rec   /= self.gradient_accumulate_every
            acc_kl    /= self.gradient_accumulate_every
            acc_total /= self.gradient_accumulate_every

            # ── gradient clipping & optimiser step ────────────────────────────
            if exists(self.max_grad_norm):
                accelerator.clip_grad_norm_(model.parameters(), self.max_grad_norm)

            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

            if self.use_lr_scheduler:
                self.scheduler.step()

            self.step += 1

            # ── logging / progress bar ────────────────────────────────────────
            if self.is_main:
                lr_now = self.opt.param_groups[0]["lr"]
                self.loss_history.append(
                    {"step": self.step, "rec": acc_rec, "kl": acc_kl, "total": acc_total}
                )
                pbar.set_postfix(
                    rec=f"{acc_rec:.4f}",
                    kl=f"{acc_kl:.4f}",
                    total=f"{acc_total:.4f}",
                    lr=f"{lr_now:.2e}",
                )
            pbar.update(1)

            # ── periodic checkpoint + optional eval ───────────────────────────
            if self.step % self.save_every == 0:
                # eval on test set
                test_metrics = self._eval_test()

                if self.is_main:
                    if exists(test_metrics):
                        self.test_loss_history.append(
                            {"step": self.step, **test_metrics}
                        )
                        pbar.write(
                            f"[step {self.step}] test →  "
                            f"rec {test_metrics['rec']:.4f}  "
                            f"kl {test_metrics['kl']:.4f}  "
                            f"total {test_metrics['total']:.4f}"
                        )
                        if test_metrics["total"] < best_test_loss:
                            best_test_loss = test_metrics["total"]
                            self.save(tag="best")
                            pbar.write(
                                f"  ↳ new best test loss {best_test_loss:.5f} — saved vae_best.pt"
                            )

                    # numbered + latest checkpoint
                    self.save(step=self.step)
                    pbar.write(f"  ↳ checkpoint saved (step {self.step})")

                    # loss plot
                    self._save_loss_plot()
                    pbar.write(
                        f"  ↳ loss plot saved → {self.results_folder / 'losses_latest.png'}"
                    )

        # ── end of training ───────────────────────────────────────────────────
        pbar.close()
        if self.is_main:
            self.save(tag="final")
            self._save_loss_plot()
            accelerator.print("Training complete. Saved vae_final.pt + final loss plot.")
        accelerator.wait_for_everyone()


def denormalise_latents(z: torch.Tensor, stats_path: str) -> torch.Tensor:
    """Apply before passing diffusion samples to vae.decode()."""
    stats = np.load(stats_path)
    mean  = torch.from_numpy(stats["mean"]).to(z)
    std   = torch.from_numpy(stats["std"]).to(z)
    return z * std + mean

@torch.no_grad()
def decode_latents(
    vae,
    z: torch.Tensor,
    batch_size: int = 8,
    device: str = "cuda",
) -> np.ndarray:
    vae = vae.to(device)
    vae.eval()

    decoded = []
    for z_batch in tqdm(z.split(batch_size), desc="Decoding latents"):
        z_batch = z_batch.to(device)
        x_hat = vae.decode(z_batch)
        decoded.append(x_hat.cpu().float())

    return torch.cat(decoded, dim=0)
