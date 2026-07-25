import argparse
import csv
import os
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from peft import get_peft_model
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import BinaryAveragePrecision

from CRISCross.Datasets import GenomicDataModule
from CRISCross.models import CRISCross, freeze_base_model, get_lora_config
from CRISCross.pretrainArtificial import PreTrainModel


# Epi features the local pretrained checkpoint was trained with (must match
# for the epi-fusion layer shape and for the datamodule to build a compatible
# epi tensor). Matches the default epi_features in pretrainArtificial.py.
PRETRAIN_EPI_FEATURES = ["H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3"]


def load_criscross_with_lora(ckpt_path: str, lora_r: int = 8, lora_alpha: int = 16):
    """Load a CRISCross backbone from a local pretrainArtificial.py (PreTrainModel) checkpoint
    and wrap it with a LoRA adapter.

    Architecture args are read from the checkpoint's saved hyperparameters so the
    rebuilt model is guaranteed to match what the checkpoint's weights expect.
    """
    ptm = PreTrainModel.load_from_checkpoint(ckpt_path, weights_only=False, map_location="cpu")
    cfg = {
        "vocab_size": 5,
        "dropout": ptm.hparams.dropout,
        "context_layers": ptm.hparams.context_layers,
        "hidden_dim": ptm.hparams.hidden_dim,
        "num_epi": ptm.hparams.num_epi,
        "output_size": 1,
        "windowsize": ptm.hparams.windowsize,
        "merge": ptm.hparams.merge,
    }
    model = CRISCross(**cfg)
    model.load_state_dict(ptm.model.state_dict())

    lora_config = get_lora_config(r=lora_r, alpha=lora_alpha)
    peft_model = get_peft_model(model, lora_config)
    freeze_base_model(peft_model)

    return peft_model, cfg


class CRISPRLoraWrapper(pl.LightningModule):
    def __init__(self, model, lr: float = 1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()
        self.auprc = BinaryAveragePrecision()

    def forward(self, target_x, off_target_x, epi, strands):
        return self.model(target_x, off_target_x, strands, epi)[0]

    def training_step(self, batch, batch_idx):
        target_x, off_target_x, epi, y, counts, strands = batch
        logits = self(target_x, off_target_x, epi, strands).squeeze(1)
        loss = self.criterion(logits, y.float())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        target_x, off_target_x, epi, y, counts, strands = batch
        logits = self(target_x, off_target_x, epi, strands).squeeze(1)
        loss = self.criterion(logits, y.float())
        preds = torch.sigmoid(logits)
        self.auprc.update(preds, y.int())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        auprc = self.auprc.compute()
        self.log("val_auprc", auprc, prog_bar=True)
        self.auprc.reset()

    def configure_optimizers(self):
        trainable = filter(lambda p: p.requires_grad, self.parameters())
        return torch.optim.AdamW(trainable, lr=self.lr, weight_decay=0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Single-guide, single-seed LoRA fine-tuning sanity check. Mirrors "
            "lora_finetuning_multiseed.py's data/model/training config exactly, "
            "minus the seed loop, so you can smoke-test the pipeline on one "
            "(guide, seed) pair before launching the full multi-guide/multi-seed sweep."
        )
    )
    parser.add_argument("--test_guide", type=str, required=True,
                        help="GuideID to hold out as the test set (e.g. sg7)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Single seed to run (default: 0)")
    parser.add_argument("--results_file", type=str, default="results/lora_sanity_check.csv",
                        help="CSV file to write the AUPRC result to")
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["gpu", "cpu"],
                        help="Device to train on. Use 'cpu' if no GPU is available.")
    parser.add_argument(
        "--pretrain_ckpt",
        type=str,
        default=(
            "RUNlogs/PretrainingArtificialTest/test_split1/"
            "ctl3_bs1024_ws23_ue6_seed0_energyFalse_hash7511e7/"
            "run_/v0/checkpoints/best_model.ckpt"
        ),
        help=(
            "Path to the local pretrainArtificial.py checkpoint (PreTrainModel .ckpt) "
            "to fine-tune from. The default assumes v0 (first run) at that hash; "
            "adjust the version folder if you've run pretraining more than once."
        ),
    )
    args = parser.parse_args()

    TEST_GUIDE = args.test_guide
    seed = args.seed

    df = pd.read_csv("datasets/TCellDataset.tsv", sep="\t")
    all_guides = sorted(df["GuideID"].unique().tolist())
    if TEST_GUIDE not in all_guides:
        raise ValueError(f"--test_guide '{TEST_GUIDE}' not found. Available: {all_guides}")

    precision = "bf16-mixed" if args.accelerator == "gpu" else "32"
    fields = ["guide_id", "seed", "val_auprc", "test_auprc", "n_test_samples"]

    results_dir = os.path.dirname(args.results_file)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[SANITY CHECK] LoRA fine-tuning: guide={TEST_GUIDE} seed={seed}")
    print(f"{'='*60}")

    pl.seed_everything(seed, workers=True)

    # ============ Load Pretrained Model + LoRA ============
    peft_model, cfg = load_criscross_with_lora(args.pretrain_ckpt, lora_r=8, lora_alpha=16)
    lightning_model = CRISPRLoraWrapper(peft_model, lr=1e-4)

    # ============ Load Data ============
    datamodule = GenomicDataModule(
        fasta_path="data/GRCh38.primary_assembly.genome.fa",
        epi_features=PRETRAIN_EPI_FEATURES,
        bw_dir="AGTensorsCL:0000624",
        df=df,
        test_guides=[TEST_GUIDE],
        val_guides=[],
        batch_size=256,
        window_size=cfg["windowsize"],
        num_samples=256 * 10,
        norm_epi=True,
    )

    # ============ Train ============
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints_lora_sanity/{TEST_GUIDE}/seed_{seed}",
        monitor="val_auprc",
        mode="max",
        filename="best-lora-model",
        save_top_k=1,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_auprc",
        mode="max",
        patience=10,
        verbose=True,
    )

    run_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_logger = TensorBoardLogger(
        save_dir="logs",
        name=f"lora_sanity_{TEST_GUIDE}",
        version=f"seed_{seed}_{run_version}",
    )

    trainer = pl.Trainer(
        max_epochs=150,
        accelerator=args.accelerator,
        precision=precision,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=tb_logger,
    )
    trainer.fit(lightning_model, datamodule)

    # ============ Test ============
    # Load the best checkpoint (highest val_auprc) rather than the final epoch weights.
    if not checkpoint_callback.best_model_path:
        raise RuntimeError(
            f"No checkpoint was saved for {TEST_GUIDE} seed {seed}. "
            "Training may have ended before the first validation epoch completed."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_ckpt = torch.load(
        checkpoint_callback.best_model_path, map_location=device, weights_only=False
    )
    lightning_model.load_state_dict(best_ckpt["state_dict"])

    lightning_model.eval()
    lightning_model.to(device)

    # ============ Save LoRA Adapter ============
    # Saved after loading the best checkpoint so the adapter reflects the best weights.
    lora_save_dir = f"lora_adapters_sanity/{TEST_GUIDE}/seed_{seed}"
    os.makedirs(lora_save_dir, exist_ok=True)
    peft_model.save_pretrained(lora_save_dir)
    print(f"LoRA adapter saved to: {lora_save_dir}")

    auprc_metric = BinaryAveragePrecision().to(device)
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for target_x, off_target_x, epi, y, counts, strands in datamodule.test_dataloader():
            target_x = target_x.to(device)
            off_target_x = off_target_x.to(device)
            epi = epi.to(device)
            y = y.to(device)
            strands = strands.to(device)

            logits = lightning_model(target_x, off_target_x, epi, strands).squeeze(1)
            probs = torch.sigmoid(logits)
            auprc_metric.update(probs, y.int())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_auprc = auprc_metric.compute().item()
    best_val_auprc = checkpoint_callback.best_model_score.item()
    print(f"[{TEST_GUIDE}] seed={seed}  val AUPRC={best_val_auprc:.4f}  test AUPRC={test_auprc:.4f}")

    # ============ Save Result ============
    new_row = {
        "guide_id": TEST_GUIDE,
        "seed": seed,
        "val_auprc": round(best_val_auprc, 4),
        "test_auprc": round(test_auprc, 4),
        "n_test_samples": len(all_probs),
    }

    existing = []
    if os.path.exists(args.results_file):
        with open(args.results_file, newline="") as f:
            existing = [
                r for r in csv.DictReader(f)
                if not (r["guide_id"] == TEST_GUIDE and int(r["seed"]) == seed)
            ]

    with open(args.results_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(new_row)

    print(f"Result saved to: {args.results_file}")

