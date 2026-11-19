"""Evaluate a trained AR ViT with FluidBench and (optionally) publish it into a FluidBench clone.

FluidBench (github.com/sp-droid/FluidBench) is used from a clone next to this
repo, without installing it (its pyproject asks for Python >= 3.13; the code
only needs h5py, numpy, torch and huggingface_hub). Its src/ is put on the path
and the script works from the clone root, where FluidBench's .gitignore covers
the downloads/ and submission/ folders it writes.

Run from the FluidFlow root:

    # login node (needs internet): download the benchmark into <fluidbench_dir>/downloads/
    PYTHONPATH=$PWD python scripts/submit_fluidbench.py --download_only

    # GPU node: evaluate, then copy into <fluidbench_dir>/submissions/ and refresh the manifests
    PYTHONPATH=$PWD python scripts/submit_fluidbench.py \
        --results_folder results/cylinder/vit_ar_p2_d8_h512_l1_0.1 --publish

The training curves get one point per saved checkpoint (see checkpoint_history).
The script never runs git: review and commit the changes in the clone yourself.
"""
import argparse
import json
import os
import runpy
import shutil
import sys
import warnings

import torch
from torch.utils.data import DataLoader

from fluidFlow.evaluation import FluidBenchWrapper
from fluidFlow.evaluation.fluidbench import REPO_ROOT, list_milestones, load_ema_weights


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_folder", default="results/cylinder/vit_ar_p2_d8_h512_l1_0.1")
    p.add_argument("--milestone", type=int, default=None, help="model-{milestone}.pt (default: latest)")
    p.add_argument("--fluidbench_dir", default="../FluidBench")
    p.add_argument("--benchmark", default="cylinder_raw")
    p.add_argument("--id", default="ar-vit", help="submission id (folder name on the site)")
    p.add_argument("--model", default="AR ViT", help="model name shown on the leaderboard")
    p.add_argument("--description", default=None, help="one phrase, <= 240 chars (default: built from the config)")
    p.add_argument("--authors", default="David Ramos")
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--representation", default="Graph", choices=["Graph", "Grid", "Patches"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_history", action="store_true", help="publish without training curves")
    p.add_argument("--download_only", action="store_true")
    p.add_argument("--publish", action="store_true",
                   help="copy the submission into <fluidbench_dir>/submissions and run its update.py")
    return p.parse_args()


def loss_name(cfg):
    l1_weight = cfg.get("l1_weight", 0.0)
    return f"MSE + {l1_weight:g}·MAE (normalised increment)" if l1_weight else "MSE (normalised increment)"


def validation_source(cfg):
    """Split used for validation by scripts/train_cylinder_ar.py (load_cylinder: valid/, else test/)."""
    data_dir = cfg["data_dir"] if os.path.isabs(cfg["data_dir"]) else os.path.join(REPO_ROOT, cfg["data_dir"])
    return data_dir, "valid" if os.path.isdir(os.path.join(data_dir, "valid")) else "test"


def default_description(cfg, cond_names, with_history):
    text = (f"Autoregressive ViT over patches of {cfg['patch_size']} mesh cells with Fourier coordinate "
            f"embedding and adaLN conditioning on {', '.join(cond_names)}, predicting the normalised increment")
    if cfg.get("l1_weight"):
        text += f", trained with MSE + {cfg['l1_weight']:g}·MAE"
    text += "."
    if with_history and validation_source(cfg)[1] == "test":
        text += " Validation curve: loss on the test cases."
    return text


@torch.inference_mode()
def checkpoint_history(wrapper, results_folder, batch_size=8):
    """FluidBench training_history with one point per saved checkpoint model-{m}.pt.

    train: the recorded per-step training loss, averaged over the steps since
        the previous checkpoint.
    val: the same loss of each checkpoint's EMA weights on the validation pairs
        of scripts/train_cylinder_ar.py (every valid_frame_stride-th frame of
        the valid/ split, or of test/ when there is no valid/ folder).

    The trainer's own validation record (test_loss_history: one-step MSE of the
    normalised fields) is in other units than the training loss, so it is only
    recomputed here as a check.
    """
    from data.load_airfoil_unsteady import make_pair_dataset
    from data.load_cylinder import load_cylinder

    cfg = wrapper.checkpoint["config"]
    model = wrapper.model.eval()
    device = wrapper.fields_mean.device
    data_dir, val_split_name = validation_source(cfg)
    dataset_train, dataset_valid, _, coefficients = load_cylinder(
        data_dir,
        cond_keys=cfg["cond_keys"],
        time_stride=cfg["time_stride"],
        max_train_samples=cfg.get("max_train_samples"),
        max_valid_samples=cfg.get("max_valid_samples"),
        max_test_samples=cfg.get("max_test_samples"),
        patch_size=cfg["patch_size"],
    )
    assert torch.allclose(coefficients["fields_mean"].reshape(-1), wrapper.fields_mean.reshape(-1).cpu()) and \
        torch.allclose(coefficients["fields_std"].reshape(-1), wrapper.fields_std.reshape(-1).cpu()), \
        "the data in data_dir is not the data the model was trained on (normalisation stats differ)"
    pairs = make_pair_dataset(dataset_valid, frame_stride=cfg["valid_frame_stride"])
    loader = DataLoader(pairs, batch_size=batch_size, shuffle=False)
    n_train, n_valid = len(dataset_train), len(dataset_valid)
    print(f"validation: {len(pairs)} pairs from {n_valid} cases of {val_split_name}/ "
          f"({n_train} training cases)")

    losses = wrapper.checkpoint["loss_history"].double()
    recorded = wrapper.checkpoint.get("test_loss_history")
    recorded = [] if recorded is None else recorded.tolist()
    every = cfg["save_and_sample_every"]
    final_milestone = int(wrapper.checkpoint["step"]) // every
    milestones = [m for m in list_milestones(results_folder) if m <= final_milestone]

    train_curve, val_curve = [], []
    print(f"{'ckpt':>4} {'step':>7} {'train':>11} {'val':>11} {'val field MSE':>13} {'(recorded)':>11}")
    for m in milestones:
        ckpt = load_ema_weights(model, os.path.join(results_folder, f"model-{m}.pt"))
        loss_sum, sq_sum, n_pairs, n_values = 0.0, 0.0, 0, 0
        for target, conds, context in loader:
            target, conds, context = target.to(device), conds.to(device), context.to(device)
            with wrapper._autocast():
                loss = model(target, classes=conds, context=context)
                pred = model.step(context, conds)
            loss_sum += loss.item() * len(target)
            sq_sum += (((pred - target.float()) ** 2) * model.node_mask).sum().item()
            n_pairs += len(target)
            n_values += model.node_mask.sum().item() * target.shape[0] * target.shape[1]
        train_curve.append(losses[(m - 1) * every:m * every].mean().item())
        val_curve.append(loss_sum / n_pairs)
        field_mse = sq_sum / n_values
        rec = f"{recorded[m - 1]:11.4e}" if m - 1 < len(recorded) else f"{'-':>11}"
        print(f"{m:>4} {int(ckpt['step']):>7} {train_curve[-1]:11.4e} {val_curve[-1]:11.4e} {field_mse:13.4e} {rec}")
    load_ema_weights(model, wrapper.checkpoint["path"])  # back to the evaluated checkpoint

    if val_split_name == "test":
        warnings.warn("there is no valid/ split: the validation curve is computed on the test cases "
                      "(not a hold-out of the training data)")
    return {
        "train_error_history": train_curve,
        "val_error_history": val_curve,
        "train_error_name": loss_name(cfg),
        "val_split": n_valid / (n_train + n_valid),
    }


def main():
    args = parse_args()
    results_folder = os.path.abspath(args.results_folder)
    fluidbench_dir = os.path.abspath(args.fluidbench_dir)
    assert os.path.isfile(os.path.join(fluidbench_dir, "src", "fluidbench", "bench.py")), \
        f"{fluidbench_dir} is not a FluidBench clone"
    sys.path.insert(0, os.path.join(fluidbench_dir, "src"))
    import fluidbench

    # fluidbench writes downloads/ and submission/ under the working directory
    os.chdir(fluidbench_dir)
    dataset_dir = fluidbench.download(args.benchmark)
    print(f"Benchmark data in {dataset_dir}")
    if args.download_only:
        return

    with open(os.path.join(fluidbench_dir, "benchmarks", f"{args.benchmark}.json")) as f:
        benchmark = json.load(f)
    wrapper = FluidBenchWrapper.from_ar_vit_results(
        results_folder, benchmark["flowVariables"], benchmark["inputScalarNames"], milestone=args.milestone)
    wrapper.to(args.device)
    print(f"channel order (model <- benchmark): {wrapper.channel_order}, condition scalars: {wrapper.cond_indices}")

    history = None if args.no_history else checkpoint_history(wrapper, results_folder)
    cfg = wrapper.checkpoint["config"]
    cond_names = [benchmark["inputScalarNames"][i] for i in wrapper.cond_indices]
    description = args.description or default_description(cfg, cond_names, history is not None)

    config = {
        "datasetId": args.benchmark,
        "id": args.id,
        "model": args.model,
        "description": description,
        "authors": args.authors,
        "year": args.year,
        "representation": args.representation,
    }
    print(f"description ({len(description)} chars): {description}")
    submission = fluidbench.run(wrapper, config, history)
    metrics = submission["metrics"]
    print(f"one-step rmse {metrics['rmse']:.4e}, rollout rmse {metrics['rolloutRmse']:.4e}, "
          f"rollout rel L2 {metrics['rolloutRelativeL2']:.4e}, params {metrics['efficiency']['parameters']}, "
          f"{metrics['efficiency']['inferenceMs']:.3f} ms/sample")

    if args.publish:
        source = os.path.join(fluidbench_dir, "submission", args.benchmark, args.id)
        published = os.path.join(fluidbench_dir, "submissions", args.benchmark, args.id)
        shutil.copytree(source, published, dirs_exist_ok=True)
        runpy.run_path(os.path.join(fluidbench_dir, "submissions", "update.py"), run_name="__main__")
        print(f"\nPublished into {published}. Review and commit in the clone:\n"
              f"  git -C {fluidbench_dir} status\n"
              f"  git -C {fluidbench_dir} add submissions\n"
              f"  git -C {fluidbench_dir} commit -m 'Add {args.model} submission for {args.benchmark}'\n"
              f"  git -C {fluidbench_dir} push -u origin HEAD")


if __name__ == "__main__":
    main()
