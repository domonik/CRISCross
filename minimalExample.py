import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAveragePrecision
from pytorch_lightning.callbacks import ModelCheckpoint

from CRISCross.models import CRISCross
from CRISCross.Datasets import GenomicDataModule
import pandas as pd


class CRISPRWrapper(pl.LightningModule):
    def __init__(self, model, lr=1e-4):
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
        preds = torch.sigmoid(logits)
        self.auprc.update(preds, y.int())
        auprc = self.auprc.compute()
        self.log("val_auprc", auprc, on_epoch=True, prog_bar=True)
        return {"val_auprc": auprc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        return optimizer


if __name__ == "__main__":
    # ============ Load Data ============
    df = pd.read_csv("datasets/TCellDataset.tsv", sep="\t")

    test_guides = [df.iloc[0]["GuideID"]]  # First guide as test
    val_guides = []  # Empty for random 20% val split

    datamodule = GenomicDataModule(
        fasta_path="data/GRCh38.primary_assembly.genome.fa",
        epi_features=["ATAC"],
        bw_dir="AGTensorsCL:0000624",
        df=df,
        test_guides=test_guides,
        val_guides=val_guides,
        batch_size=256,
        window_size=512,
        num_samples=256 * 10,
        norm_epi=True
    )

    # ============ Load Pre-trained Model ============
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

    model = CRISCross(**cfg)
    model.load_state_dict(torch.hub.load_state_dict_from_url(
        "https://huggingface.co/domonik/criscross-atac/resolve/main/model.pt"
    ))

    # Wrap in Lightning module
    lightning_model = CRISPRWrapper(model, lr=1e-4)
    lightning_model.train()

    # ============ Train ============
    checkpoint_callback = ModelCheckpoint(
        monitor="val_auprc",
        mode="max",
        filename="best-model",
        save_top_k=1
    )

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        precision="bf16-mixed",
        callbacks=[checkpoint_callback]
    )
    trainer.fit(lightning_model, datamodule)

    # ============ Load Best Model & Predict ============
    best_checkpoint = trainer.checkpoint_callback.best_model_path
    print(f"Loading best checkpoint: {best_checkpoint}")

    # Load wrapped model from checkpoint
    checkpoint = torch.load(best_checkpoint, map_location="cuda" if torch.cuda.is_available() else "cpu")
    best_model = CRISPRWrapper(lightning_model.model, lr=1e-4)
    best_model.load_state_dict(checkpoint["state_dict"])
    best_model.to("cuda")
    print("loaded checkpoint")

    # Get predictions from test dataloader (on GPU)
    all_probs = []
    all_labels = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        for target_x, off_target_x, epi, y, counts, strands in datamodule.test_dataloader():
            target_x = target_x.to(device)
            off_target_x = off_target_x.to(device)
            epi = epi.to(device)
            y = y.to(device)
            counts = counts.to(device)
            strands = strands.to(device)

            logits = best_model(target_x, off_target_x, epi, strands).squeeze(1)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    print(f"Total predictions: {len(all_probs)}")
    print(f"Sample predictions: {all_probs[:5]}")
    print(f"Sample labels: {all_labels[:5]}")
    print("Success")
