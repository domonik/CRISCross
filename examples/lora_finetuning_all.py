import argparse
import csv
import os

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


def load_criscross_with_lora(pretrained_url: str, cfg: dict, lora_r: int = 8, lora_alpha: int = 16):
    """Load a pretrained CRISCross model and wrap it with LoRA adapters.

    Steps:
      1. Instantiate CRISCross with the given cfg.
      2. Download and load pretrained weights from pretrained_url.
      3. Wrap with LoRA using get_peft_model().
      4. Freeze all non-LoRA parameters so only adapters are trained.

    Args:
        pretrained_url: URL to a .pt state_dict file (e.g. on HuggingFace Hub).
        cfg: Dict of CRISCross constructor kwargs (vocab_size, hidden_dim, etc.).
        lora_r: LoRA rank. Default: 8.
        lora_alpha: LoRA scaling factor. Default: 16.

    Returns:
        peft.PeftModel wrapping the CRISCross base model with LoRA adapters applied.
    """
    model = CRISCross(**cfg)
    state_dict = torch.hub.load_state_dict_from_url(pretrained_url)
    model.load_state_dict(state_dict)

    lora_config = get_lora_config(r=lora_r, alpha=lora_alpha)
    peft_model = get_peft_model(model, lora_config)
    freeze_base_model(peft_model)

    return peft_model


class CRISPRLoraWrapper(pl.LightningModule):
    """PyTorch Lightning wrapper for LoRA fine-tuning of CRISCross.

    Identical interface to CRISPRWrapper in minimalExample.py, but the optimizer
    is restricted to trainable (LoRA) parameters only.
    """

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
        auprc = self.auprc.compute()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_auprc", auprc, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_auprc": auprc}

    def configure_optimizers(self):
        # Only pass parameters that require gradients (i.e. LoRA adapter weights).
        trainable = filter(lambda p: p.requires_grad, self.parameters())
        return torch.optim.AdamW(trainable, lr=self.lr, weight_decay=0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_guide", type=str, required=True,
                        help="GuideID to hold out as the test set (e.g. sg7)")
    parser.add_argument("--results_file", type=str, default="results/loso_results.csv",
                        help="CSV file to append per-guide AUPRC results")
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["gpu", "cpu"],
                        help="Device to train on. Use 'cpu' if no GPU is available.")
    args = parser.parse_args()

    TEST_GUIDE = args.test_guide
    PRETRAINED_URL = "https://huggingface.co/domonik/criscross-atac/resolve/main/model.pt"
    LORA_SAVE_DIR = f"lora_adapters/{TEST_GUIDE}"

    cfg = {
        "vocab_size": 5,
        "dropout": 0.2,
        "context_layers": 3,
        "hidden_dim": 512,
        "num_epi": 1,
        "output_size": 1,
        "windowsize": 512,
        "merge": "early",
    }

    # ============ Load Data ============
    df = pd.read_csv("datasets/TCellDataset.tsv", sep="\t")

    all_guides = sorted(df["GuideID"].unique().tolist())
    if TEST_GUIDE not in all_guides:
        raise ValueError(f"--test_guide '{TEST_GUIDE}' not found. Available: {all_guides}")

    datamodule = GenomicDataModule(
        fasta_path="data/GRCh38.primary_assembly.genome.fa",
        epi_features=["ATAC"],
        bw_dir="AGTensorsCL:0000624",
        df=df,
        test_guides=[TEST_GUIDE],
        val_guides=[],           # empty list → random 20% of train rows used as val
        batch_size=256,
        window_size=512,
        num_samples=256 * 10,
        norm_epi=True,
    )

    # ============ Load Pretrained Model + LoRA ============
    peft_model = load_criscross_with_lora(PRETRAINED_URL, cfg, lora_r=8, lora_alpha=16)
    peft_model.print_trainable_parameters()

    lightning_model = CRISPRLoraWrapper(peft_model, lr=1e-4)

    # ============ Train ============
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{TEST_GUIDE}",
        monitor="val_auprc",
        mode="max",
        filename="best-lora-model",
        save_top_k=1,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_auprc",
        mode="max",
        patience=5,
        verbose=True,
    )

    tb_logger = TensorBoardLogger(save_dir="logs", name=f"lora_{TEST_GUIDE}")

    precision = "bf16-mixed" if args.accelerator == "gpu" else "32"

    trainer = pl.Trainer(
        max_epochs=70,
        accelerator=args.accelerator,
        precision=precision,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=tb_logger,
    )
    trainer.fit(lightning_model, datamodule)

    # ============ Save LoRA Adapter ============
    os.makedirs(LORA_SAVE_DIR, exist_ok=True)
    peft_model.save_pretrained(LORA_SAVE_DIR)
    print(f"LoRA adapter saved to: {LORA_SAVE_DIR}")

    # ============ Test ============
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lightning_model.eval()
    lightning_model.to(device)

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
    print(f"[{TEST_GUIDE}] Test AUPRC: {test_auprc:.4f}")

    # ============ Save Result ============
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    best_val_auprc = checkpoint_callback.best_model_score.item()

    fields = ["guide_id", "val_auprc", "test_auprc", "n_test_samples"]
    new_row = {
        "guide_id": TEST_GUIDE,
        "val_auprc": round(best_val_auprc, 4),
        "test_auprc": round(test_auprc, 4),
        "n_test_samples": len(all_probs),
    }

    # Read existing rows, drop any previous entry for this guide, then write back.
    existing = []
    if os.path.exists(args.results_file):
        with open(args.results_file, newline="") as f:
            existing = [r for r in csv.DictReader(f) if r["guide_id"] != TEST_GUIDE]

    with open(args.results_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(new_row)
    print(f"Result saved to: {args.results_file}")

