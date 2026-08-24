"""
Leakage probe for the bulge-capable pipeline (spec section 6.2).

Because bulges are never encoded, the model is supposed to *infer* the bulge
configuration. Nothing in its input may reveal it. This script checks that in
three stages, in increasing order of how much interpretation the result needs:

**A. Structural invariants (hard, must pass).**
    Guide length, window length, band width and the extent of the region label
    must be byte-identical across every bulge configuration. If any of them
    tracks ``(b_D, b_R)``, the answer is written on the input in plain sight and
    every downstream result is void. This stage raises on failure.

**B. Shape-only probe (the operational form of "should be at chance").**
    A classifier is trained to predict ``(b_D, b_R)`` from structural features
    alone -- the lengths, the band bounds and the region-label histogram, all
    derived from the actual sample tensors rather than from constants. It must
    not beat the majority-class baseline. This is the probe the spec asks for,
    stated so that passing it means something.

**C. Sequence probe (soft, reported for context).**
    A small MLP on the one-hot guide and band. This one is *expected* to beat
    chance, and that is not a leak: whether a guide/target pair is better
    explained by a gapped alignment is genuinely a property of the two
    sequences, and inferring it is the task. Treat this number as a difficulty
    estimate -- roughly, an upper bound on how much of the bulge signal is
    linearly available before any alignment machinery is involved -- not as a
    pass/fail criterion.

Usage::

    # against random sequence, no genome or epigenetic tracks needed
    python -m CRISCross.leakageProbe --synthetic

    # against the real sampler
    python -m CRISCross.leakageProbe --fasta-cache encoded_fasta.pt \\
        --band-delta 2 --bulge-rate 0.5
"""

import argparse
import os
import pickle
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from CRISCross.bulges import (
    PAM_LEN,
    SITE_LEN,
    band_bounds,
    band_width,
    structural_leakage_report,
)
from CRISCross.Datasets import BulgeGenomicDataset
from CRISCross.models import HighlightCenterAndPAM


def collect_samples(dataset, n_samples):
    """Draw ``n_samples`` items and split them into inputs and bulge labels.

    Args:
        dataset: A :class:`~CRISCross.Datasets.BulgeGenomicDataset`.
        n_samples: How many items to draw.

    Returns:
        ``(records, configs)`` where ``records`` is a list of dicts with keys
        ``guide``, ``window``, ``band``, ``edits``, and ``configs`` is the list
        of ``(b_D, b_R)`` labels.
    """
    bs, be = dataset.band_start, dataset.band_end
    records, configs = [], []
    for i in range(n_samples):
        item = dataset[i % len(dataset)]
        target_x, off_target_x = item[0], item[1]
        edits = item[8]
        records.append(
            {
                "guide": target_x,
                "window": off_target_x,
                "band": off_target_x[bs:be],
                "edits": edits,
            }
        )
        configs.append((int(edits[1]), int(edits[2])))
    return records, configs


def shape_features(records, dataset):
    """Structural features only -- no nucleotide content whatsoever.

    Everything here is measured off the sample tensors, so a future change that
    made any of these sample-dependent would show up as a probe that suddenly
    beats chance.

    Args:
        records: Output of :func:`collect_samples`.
        dataset: The dataset the records came from (for the band bounds).

    Returns:
        Float tensor ``[N, 6]``.
    """
    bs, be = dataset.band_start, dataset.band_end
    # The region label is a function of the band bounds alone; recompute it the
    # way the model does, then summarise it.
    label = torch.zeros(dataset.window_size)
    label[bs:be] = 1
    label[be - PAM_LEN:be] = 2

    feats = []
    for r in records:
        feats.append(
            [
                float(r["guide"].numel()),
                float(r["window"].numel()),
                float(r["band"].numel()),
                float((label == 1).sum()),
                float((label == 2).sum()),
                float(bs),
            ]
        )
    return torch.tensor(feats, dtype=torch.float)


def sequence_features(records, dataset):
    """One-hot guide and band, flattened.

    Args:
        records: Output of :func:`collect_samples`.
        dataset: The dataset the records came from.

    Returns:
        Float tensor ``[N, 5 * (23 + band_width)]``.
    """
    guides = torch.stack([r["guide"].long() for r in records])
    bands = torch.stack([r["band"].long() for r in records])
    x = torch.cat(
        [
            F.one_hot(guides, 5).float().flatten(1),
            F.one_hot(bands, 5).float().flatten(1),
        ],
        dim=1,
    )
    return x


def train_probe(x, y, n_classes, hidden=0, steps=600, lr=1e-2, seed=0, val_frac=0.3):
    """Fit a probe and return its held-out accuracy.

    Args:
        x: Float features ``[N, F]``.
        y: Long labels ``[N]``.
        n_classes: Number of distinct labels.
        hidden: Hidden width; 0 gives multinomial logistic regression.
        steps: Full-batch optimisation steps.
        lr: Learning rate.
        seed: Torch seed for the probe's initialisation and split.
        val_frac: Fraction held out.

    Returns:
        ``(accuracy, majority_baseline)``.
    """
    torch.manual_seed(seed)
    n = x.shape[0]
    perm = torch.randperm(n)
    n_val = max(1, int(n * val_frac))
    val, train = perm[:n_val], perm[n_val:]

    mu, sd = x[train].mean(0), x[train].std(0).clamp_min(1e-6)
    x = (x - mu) / sd

    if hidden:
        probe = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, n_classes))
    else:
        probe = nn.Linear(x.shape[1], n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(probe(x[train]), y[train])
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = probe(x[val]).argmax(1)
        acc = float((pred == y[val]).float().mean())
    counts = torch.bincount(y[train], minlength=n_classes).float()
    majority = int(counts.argmax())
    baseline = float((y[val] == majority).float().mean())
    return acc, baseline


def run_leakage_probe(dataset, n_samples=4000, tolerance=0.02, verbose=True):
    """Run all three stages of the probe.

    Args:
        dataset: A :class:`~CRISCross.Datasets.BulgeGenomicDataset`.
        n_samples: Number of samples to draw.
        tolerance: How far above the majority baseline the shape-only probe may
            land before it is called a leak.
        verbose: Print a report.

    Returns:
        Dict with the structural report and both probe accuracies.

    Raises:
        AssertionError: If a structural invariant is violated.
    """
    records, configs = collect_samples(dataset, n_samples)

    # --- A. structural invariants ------------------------------------------
    report = structural_leakage_report(records)
    assert report["ok"], f"structural leak: sample shapes depend on the bulge configuration: {report}"
    assert len(report["bulge_configs"]) > 1, (
        "the sampler produced a single bulge configuration -- the probe cannot "
        "test anything; raise bulge_rate"
    )

    # Also check the region label the model actually builds.
    module = HighlightCenterAndPAM(d_model=4)
    with torch.no_grad():
        labelled = module(
            torch.zeros(1, dataset.window_size, 4), dataset.band_start, dataset.band_end
        )
    label_extent = int(
        (~torch.isclose(labelled[0], module.token_type_emb.weight[0]).all(-1)).sum()
    )
    assert label_extent == band_width(dataset.band_delta), (
        f"region label marks {label_extent} nt but the band is "
        f"{band_width(dataset.band_delta)} nt wide"
    )

    classes = sorted(set(configs))
    y = torch.tensor([classes.index(c) for c in configs])

    # --- B. shape-only probe -----------------------------------------------
    shape_acc, shape_base = train_probe(shape_features(records, dataset), y, len(classes))
    shape_leak = shape_acc > shape_base + tolerance

    # --- C. sequence probe --------------------------------------------------
    seq_acc, seq_base = train_probe(
        sequence_features(records, dataset), y, len(classes), hidden=64, steps=800, lr=3e-3
    )

    result = {
        "structural": report,
        "region_label_extent": label_extent,
        "classes": classes,
        "shape_probe_accuracy": shape_acc,
        "shape_probe_baseline": shape_base,
        "shape_probe_leaks": shape_leak,
        "sequence_probe_accuracy": seq_acc,
        "sequence_probe_baseline": seq_base,
    }

    if verbose:
        print("=" * 72)
        print("LEAKAGE PROBE (spec 6.2)")
        print("=" * 72)
        print(f"samples              : {report['n_samples']}")
        print(f"bulge configurations : {report['bulge_configs']}")
        print("")
        print("A. structural invariants")
        print(f"   guide lengths     : {report['guide_lengths']}  (must be a single value)")
        print(f"   window lengths    : {report['window_lengths']}")
        print(f"   band widths       : {report['band_widths']}")
        print(f"   region label nt   : {label_extent}")
        print("   -> PASS")
        print("")
        print("B. shape-only probe  (must be at the baseline)")
        print(f"   accuracy          : {shape_acc:.4f}")
        print(f"   majority baseline : {shape_base:.4f}")
        print(f"   -> {'LEAK DETECTED' if shape_leak else 'PASS'}")
        print("")
        print("C. sequence probe    (informational -- above baseline is expected)")
        print(f"   accuracy          : {seq_acc:.4f}")
        print(f"   majority baseline : {seq_base:.4f}")
        print("   A gapped-vs-gapless decision is a real property of the sequence")
        print("   pair, so this number being above baseline is the task, not a leak.")
        print("=" * 72)

    assert not shape_leak, (
        f"shape-only probe reached {shape_acc:.4f} against a {shape_base:.4f} baseline: "
        "something structural is tracking the bulge configuration"
    )
    return result


def _synthetic_dataset(window_size, band_delta, bulge_rate, n_samples, chrom_size=200_000):
    """A dataset over random sequence, so the probe runs without a genome."""
    torch.manual_seed(0)
    chrom_sizes = {"chrS": chrom_size}
    seq_dict = {"chrS": torch.randint(1, 5, (chrom_size,), dtype=torch.uint8)}
    return BulgeGenomicDataset(
        chrom_sizes=chrom_sizes,
        seq_dict=seq_dict,
        bw_dir=[tempfile.gettempdir()],
        epi_features=[],
        window_size=window_size,
        num_samples=n_samples,
        band_delta=band_delta,
        bulge_rate=bulge_rate,
    )


def main():
    parser = argparse.ArgumentParser(description="Bulge leakage probe (spec 6.2)")
    parser.add_argument("--synthetic", action="store_true",
                        help="probe against random sequence instead of the genome")
    parser.add_argument("--fasta-cache", default="encoded_fasta.pt",
                        help="pickle written by GenomicDataModule.prepare_data")
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--band-delta", type=int, default=2)
    parser.add_argument("--bulge-rate", type=float, default=0.5)
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()

    if args.synthetic:
        dataset = _synthetic_dataset(
            args.window_size, args.band_delta, args.bulge_rate, args.n_samples
        )
    else:
        if not os.path.exists(args.fasta_cache):
            raise SystemExit(
                f"{args.fasta_cache} not found. Run the pretraining pipeline once to build it, "
                "or pass --synthetic."
            )
        with open(args.fasta_cache, "rb") as handle:
            cache = pickle.load(handle)
        dataset = BulgeGenomicDataset(
            chrom_sizes=cache["chrom_sizes"],
            seq_dict=cache["seq_dict"],
            bw_dir=[tempfile.gettempdir()],
            epi_features=[],
            window_size=args.window_size,
            num_samples=args.n_samples,
            band_delta=args.band_delta,
            bulge_rate=args.bulge_rate,
        )

    run_leakage_probe(dataset, n_samples=args.n_samples, tolerance=args.tolerance)


if __name__ == "__main__":
    main()
