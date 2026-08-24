#!/usr/bin/env python3
"""
Emit the gapless/bulged ablation sweep as a SLURM-array config (spec section 3.3).

Writes a JSON list of pretraining configs -- one per bulge rate, times one per
seed -- in the format ``CRISCross.pretrainArtificial`` reads via ``--config``,
so it drops straight into ``scripts_sh/run_pretrain_array.sh``::

    python examples/make_bulge_sweep_config.py --out configs/bulge_sweep.json
    python -c "import json;print(len(json.load(open('configs/bulge_sweep.json')))-1)"
    sbatch --array=0-<that number> scripts_sh/run_pretrain_array.sh

The 0 % endpoint is the mismatch-only control: ``bulge_rate=0.0`` makes the
sampler emit nothing but gapless pairs. ``band_delta`` is held constant along
the sweep so that only the bulge rate varies. Pass ``--include-legacy-control``
for an extra ``band_delta=0`` run, which is the separate Phase 1 control that
answers whether the widened band costs anything on mismatch-only data. The
100 % endpoint doubles as the bulge-specialist upper bound.

This is the sweep itself, i.e. a Phase 2 deliverable; it lives here because the
Phase 1 pipeline is what makes it a one-liner.
"""

import argparse
import json
import os

DEFAULT_RATES = [0.0, 0.05, 0.2, 0.5, 1.0]


def build_configs(rates, seeds, band_delta, base, include_legacy_control=False):
    """One config per (rate, seed), plus an optional pre-extension control.

    ``band_delta`` is held constant across the whole sweep, including the 0 %
    point, so that the only thing varying along the curve is the bulge rate. A
    0 % point run at ``band_delta=0`` would differ from its neighbours in two
    ways at once and would not be a valid control for them.

    ``include_legacy_control`` adds one extra config at ``bulge_rate=0`` and
    ``band_delta=0``. That one is a different control -- it answers "does the
    wider PAM-anchored band cost anything on mismatch-only data?", which is the
    Phase 1 success criterion, not a point on the sweep.
    """
    configs = []
    for rate in rates:
        for seed in seeds:
            cfg = dict(base)
            cfg["seed"] = seed
            cfg["bulge_rate"] = rate
            cfg["band_delta"] = band_delta
            cfg["experiment"] = f"{base['experiment']}/bulge{int(round(rate * 100))}pct"
            configs.append(cfg)
    if include_legacy_control:
        for seed in seeds:
            cfg = dict(base)
            cfg["seed"] = seed
            cfg["bulge_rate"] = 0.0
            cfg["band_delta"] = 0
            cfg["experiment"] = f"{base['experiment']}/legacyControl"
            configs.append(cfg)
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", default="configs/bulge_sweep.json")
    parser.add_argument("--rates", type=float, nargs="+", default=DEFAULT_RATES)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--band-delta", type=int, default=2)
    parser.add_argument("--windowsize", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--bw-dir", nargs="+", default=["AGTensorsCL:0000624"])
    parser.add_argument("--experiment", default="PretrainingBulgeSweep")
    parser.add_argument("--use-energy", action="store_true",
                        help="add the hybrid-energy head; it is masked on bulged samples")
    parser.add_argument("--include-legacy-control", action="store_true",
                        help="also emit a band_delta=0 mismatch-only run (the Phase 1 control)")
    args = parser.parse_args()

    base = {
        "batch_size": args.batch_size,
        "context_layers": 3,
        "hidden_dim": 512,
        "embed_size": 32,
        "dropout": 0.2,
        "epi_features": [
            "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3",
        ],
        "lr": 1e-4,
        "patience": 100000,
        "split": 1,
        "experiment": args.experiment,
        "regression": False,
        "windowsize": args.windowsize,
        "merge": "early",
        "model_type": "crosscrispr",
        "use_energy": bool(args.use_energy),
        "bw_dir": list(args.bw_dir),
        "epi_mode": "np",
        "atac_features": ["ATAC"],
        "atac_weight": 0.1,
        "accumulate_grad_batches": 48,
        "max_steps": args.max_steps,
        # Sampler defaults, spelled out so the ablation is self-documenting.
        "mismatch_span": "site",
        "mismatch_base_factor": 3,
        "position_tilt": 0.0,
        "canonicalise_homopolymers": True,
    }

    configs = build_configs(
        args.rates, args.seeds, args.band_delta, base, args.include_legacy_control
    )
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(configs, handle, indent=2)

    print(f"wrote {len(configs)} configs to {args.out}")
    print(f"  bulge rates : {args.rates}")
    print(f"  seeds       : {args.seeds}")
    print(f"  band_delta  : {args.band_delta} (constant across the sweep)")
    print(f"sbatch --array=0-{len(configs) - 1} scripts_sh/run_pretrain_array.sh")


if __name__ == "__main__":
    main()
