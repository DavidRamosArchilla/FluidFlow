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
        --results_folder results/cylinder/vit_ar_p8_d8_h512_first_test --publish

The script never runs git: review and commit the changes in the clone yourself.
"""
import argparse
import json
import os
import runpy
import shutil
import sys
import warnings

import h5py
import numpy as np
import torch

from fluidFlow.evaluation import FluidBenchWrapper

DEFAULT_DESCRIPTION = ("Autoregressive ViT over patches of 8 mesh cells with Fourier coordinate embedding "
                       "and adaLN conditioning on Re, predicting the normalised increment.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_folder", default="results/cylinder/vit_ar_p8_d8_h512_first_test")
    p.add_argument("--milestone", type=int, default=None, help="model-{milestone}.pt (default: latest)")
    p.add_argument("--fluidbench_dir", default="../FluidBench")
    p.add_argument("--benchmark", default="cylinder_Re41_raw")
    p.add_argument("--id", default="ar-vit", help="submission id (folder name on the site)")
    p.add_argument("--model", default="AR ViT", help="model name shown on the leaderboard")
    p.add_argument("--description", default=DEFAULT_DESCRIPTION)
    p.add_argument("--authors", default="David Ramos")
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--representation", default="Graph", choices=["Graph", "Grid", "Patches"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--history_points", type=int, default=20, help="points of the published training curve")
    p.add_argument("--val_fraction", type=float, default=0.1, help="tail of each training case for the val loss")
    p.add_argument("--no_history", action="store_true", help="publish without training curves")
    p.add_argument("--download_only", action="store_true")
    p.add_argument("--publish", action="store_true",
                   help="copy the submission into <fluidbench_dir>/submissions and run its update.py")
    return p.parse_args()


@torch.inference_mode()
def tail_validation_loss(wrapper, train_dir, val_fraction, batch_size=8):
    """Training loss (normalized-increment MSE) of the wrapped model on the last pairs of each training case."""
    model = wrapper.model.eval()
    case_paths = sorted((p for p in os.listdir(train_dir) if p.endswith(".h5") and p[:-3].isdigit()),
                        key=lambda p: int(p[:-3]))
    total, count = 0.0, 0
    for name in case_paths:
        with h5py.File(os.path.join(train_dir, name), "r") as case_file:
            scalars, fields = torch.from_numpy(case_file["scalars"][:]), torch.from_numpy(case_file["fields"][:])
        n_pairs = len(fields) - 1
        start = n_pairs - max(1, round(n_pairs * val_fraction))
        for i in range(start, n_pairs, batch_size):
            j = min(i + batch_size, n_pairs)
            context, target = wrapper._normalize(fields[i:j]), wrapper._normalize(fields[i + 1:j + 1])
            with wrapper._autocast():
                loss = model(target, classes=wrapper._conds(scalars[i:j]), context=context)
            total += loss.item() * (j - i)
            count += j - i
    return total / count


def training_history(wrapper, train_dir, points, val_fraction):
    """FluidBench training_history from the checkpoint loss history.

    The train curve is the real per-step loss averaged into ``points`` chunks.
    Only the final checkpoint is available, so the validation curve repeats a
    single value: the final model's loss on the tail of the training cases,
    which were not held out during training. It is a placeholder.
    """
    losses = wrapper.checkpoint["loss_history"].double().numpy()
    train_curve = [float(chunk.mean()) for chunk in np.array_split(losses, min(points, len(losses)))]
    val_loss = tail_validation_loss(wrapper, train_dir, val_fraction)
    warnings.warn(f"the validation curve is a placeholder: the final val loss {val_loss:.3e} is repeated "
                  f"{len(train_curve)} times, and those pairs were not held out during training")
    return {
        "train_error_history": train_curve,
        "val_error_history": [val_loss] * len(train_curve),
        "train_error_name": "MSE (normalised increment)",
        "val_split": val_fraction,
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

    history = None
    if not args.no_history:
        history = training_history(wrapper, os.path.join(dataset_dir, "train"), args.history_points, args.val_fraction)

    config = {
        "datasetId": args.benchmark,
        "id": args.id,
        "model": args.model,
        "description": args.description,
        "authors": args.authors,
        "year": args.year,
        "representation": args.representation,
    }
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
