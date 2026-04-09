# CRISCross

A PyTorch Lightning-based deep learning framework for predicting CRISPR off-target effects using transformer-based architectures that combine sequence information with epigenetic features.

## Quick Start

This guide shows how to fine-tune the CRISCross model with epigenetic features on the TCell dataset.

**For a complete end-to-end example, run `minimalExample.py`** - it handles data loading, fine-tuning, checkpoint saving, and predictions automatically.

### Step 1: Download Genome FASTA

```bash
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/GRCh38.primary_assembly.genome.fa.gz
gunzip GRCh38.primary_assembly.genome.fa.gz
mkdir data
mv GRCh38.primary_assembly.genome.fa data/
```

### Step 2: Download AlphaGenome ATAC Tracks for T Cells (CL:0000624)

Note: Requires `AGAPIKEY` env variable to be set. 

```bash
# Install the package first
pip install -e .

# Download ATAC tracks for T cells Note the output-dir matches the used dataframes epiDir column
criscross-download-ag --ontology CL:0000624 --features ATAC --output-dir AGTensorsCL:0000624
```

### Step 3: Examine the TCell Dataset

```python
import pandas as pd

df = pd.read_csv("datasets/TCellDataset.tsv", sep="\t")
print(df.head())
print(f"Total samples: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}")
```

### Step 4: Setup Data Module

```python
import torch
from CRISCross.models import CRISCross
from CRISCross.Datasets import GenomicDataModule

df = pd.read_csv("datasets/TCellDataset.tsv", sep="\t")

# Pick one test guide from the dataset
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
    num_samples=256 * 10,  # batch_size * 10
    norm_epi=True
)
```

### Step 5: Load Pre-trained Model and Fine-tune

You need to wrap the CRISCross model in a PyTorch Lightning module to use the trainer. Here's a minimal example:

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAveragePrecision
from pytorch_lightning.callbacks import ModelCheckpoint

from CRISCross.models import CRISCross

# ============ Define Lightning Module ============

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
        # Simple constant LR (no warmup)
        return optimizer


# ============ Load Pre-trained Model ============
cfg = {
    "vocab_size": 5,
    "dropout": 0.2,
    "context_layers": 3,
    "hidden_dim": 512,
    "num_epi": 1,  # One epigenetic feature (ATAC)
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
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="val_auprc",
    mode="max",
    filename="best-model",
    save_top_k=1
)

trainer = pl.Trainer(
    max_epochs=10,
    accelerator="gpu",
    precision="bf16-mixed",
    callbacks=[checkpoint_callback]
)

trainer.fit(lightning_model, datamodule)
```

### Step 6: Make Predictions

Load the best checkpoint and make predictions on GPU:

```python
import torch
from CRISCross.models import CRISCross

# Load best checkpoint from training
best_checkpoint = trainer.checkpoint_callback.best_model_path
checkpoint = torch.load(best_checkpoint, map_location="cuda" if torch.cuda.is_available() else "cpu")

# Create model and load state dict
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
model.load_state_dict(checkpoint["state_dict"])

# Wrap in Lightning module and move to CUDA
from pytorch_lightning import LightningModule

class CRISPRWrapper(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, target_x, off_target_x, epi, strands):
        return self.model(target_x, off_target_x, strands, epi)[0]

best_model = CRISPRWrapper(model)
best_model = best_model.to("cuda")
best_model.eval()

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
```

### Sequence-Only Model

For sequence-only fine-tuning (no epigenetic features), modify `minimalExample.py`:

1. Change `epi_features=[]` in datamodule
2. Change `cfg` to use `num_epi=0` and `merge=None`
3. Load the sequence-only model:

```python
model.load_state_dict(torch.hub.load_state_dict_from_url(
    "https://huggingface.co/domonik/criscross-seq-only/resolve/main/model.pt"
))
```

## Architecture Overview

CRISCross uses a transformer-based architecture with:

- **Self-attention layers**: Separate attention for target (guide) and off-target sequences
- **Cross-attention layers**: Bidirectional attention between target and off-target PAM-proximal regions
- **Positional encodings**: BERT-style learned absolute positional embeddings
- **Epigenetic feature fusion**: Optional early fusion of epigenetic marks (ATAC, DNASE, histone modifications, RNA-seq)

### Key Components

| Component | Description |
|-----------|-------------|
| `CRISCross` | Main transformer model (`CRISCross/models.py`) |
| `GenomicDataModule` | PyTorch Lightning data module (`CRISCross/Datasets.py`) |
| `FineTuningGenomicDataset` | Dataset class for dataframe-based fine-tuning |

## Prerequisites

### Python Dependencies

```bash
pip install torch>=2.0 numpy huggingface_hub pytorch_lightning pyBigWig biopython pandas alphagenome
```

### Environment Variables

```bash
export AGAPIKEY="your_alpha_genome_api_key"  # For downloading AlphaGenome tracks
export TMPDIR="/path/to/tmp"  # For temporary data storage
```

## Data Requirements

### Dataframe Schema

The `GenomicDataModule` expects a pandas DataFrame with the following **required columns**:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Guide_sequence` | str | Guide sequence: **23 nucleotides** (PAM-proximal center) | `"GCTAGCTAGCTAGCTAGCTAGCTAG"` |
| `chr` | str | Chromosome name | `"chr1"` or `"1"` |
| `start` | int | Start position of target region (0-based) | `12345678` |
| `end` | int | End position of target region | `12345724` |
| `Strand` | str | DNA strand: `"+"` or `"-"` | `"+"` |
| `label` | int/float | Target label for fine-tuning | `0` or `1` for binary, float for regression |
| `GuideID` | str | Unique identifier for train/val/test split |


### Optional Columns

| Column | Type | Description |
|--------|------|-------------|
| `epiDir` | str | Path to epigenetic features directory (for per-sample epi files) |

### About 23nt Sequences and Genomic Coordinates

The CRISCross model processes **23-nucleotide sequences** for the guide/target region. Here's how the dataframe coordinates map to the model input:

- **`Guide_sequence`**: Exactly **23 nucleotides** - the PAM-proximal center region of the guide
- **`start` and `end`**: Genomic coordinates defining a **23nt span** (`end = start + 23`, with `end` being exclusive)
- **`window_size=512`**: The Datamodule automatically extends the 23nt guide sequence to a 512nt off-target window by extracting flanking genomic sequence


The Datamodule handles all sequence extensions internally - you only need to provide the 23nt guide sequence and candidate off-target genomic coordinates.

### Example: Loading TCellDataset.tsv

```python
import pandas as pd

# Load the provided TCellDataset
df = pd.read_csv("datasets/TCellDataset.tsv", sep="\t")
print(df.head())
#    Guide_sequence    chr   start   end Strand  label   GuideID
# 0   GCTAGCTAGCTAGCTAGCTAGCTAG  chr1  100000  100046    +      0  guide_001
# 1   TTTAAAAAAAACCCCCGGGGGTAGG  chr1  200000  200046    -      1  guide_002
# ...

# Check sequence lengths (should be 23 for Guide_sequence)
print(f"Guide_sequence length: {len(df['Guide_sequence'].iloc[0])}")  # 23

# Check label distribution
print(df["label"].value_counts())
```

### Dataframe Example (23nt sequences)

```python
import pandas as pd

# Create a minimal dataframe with correct 23nt sequences
df = pd.DataFrame({
    "Guide_sequence": [
        "GCTAGCTAGCTAGCTAGCTAGCTAG",  # 23nt guide sequence
        "TTTTAAAAAAAACCCCCGGGGGTAG",  # 23nt guide sequence
    ],
    "chr": ["chr1", "chr2"],
    "start": [100000, 200000],
    "end": [100023, 200023],  # start + 23 for 23nt span on both sides
    "Strand": ["+", "-"],
    "label": [0, 1],  # binary classification
    "GuideID": ["guide_001", "guide_002"]
})

# Verify all sequences are 23nt
assert all(len(seq) == 23 for seq in df["Guide_sequence"])
```

## Preparing Epigenetic Features

### Using AlphaGenome Tracks

Download AlphaGenome tracks using the provided CLI tool:

```bash
criscross-download-ag --ontology CL:0000624 \
    --features ATAC DNASE CHIP_HISTONE RNA_SEQ \
    --output-dir AGTensors_CL0000624
```

This downloads numpy memmap files (`.npy`) for each chromosome and feature.

### File Structure

Expected directory structure:

```
data/alphagenome/
├── ATAC_chr1.npy
├── ATAC_chr2.npy
├── ...
├── DNASE_chr1.npy
├── DNASE_chr2.npy
└── ...
```

Each `.npy` file is a numpy memmap with shape `(genome_length, 1)` or `(genome_length, n_channels)`.

## Using the GenomicDataModule

### Basic Usage

```python
from CRISCross.Datasets import GenomicDataModule

datamodule = GenomicDataModule(
    fasta_path="data/hg38.fa",
    epi_features=["ATAC", "DNASE"],  # List of features (use [] for seq-only)
    bw_dir="data/alphagenome",
    df=my_dataframe,
    test_guides=[],   # GuideIDs for test set
    val_guides=[],    # GuideIDs for val set (empty = random 20% split)
    batch_size=32,
    window_size=512,
    num_samples=10000,  # Number of samples for oversampling
    norm_epi=True       # Auto-normalize epigenetic features
)
```

### Train/Val/Test Split

```python
# Option 1: Manual split by GuideID
test_guides = ["guide_001", "guide_002"]
val_guides = ["guide_003"]

datamodule = GenomicDataModule(
    df=df,
    test_guides=test_guides,
    val_guides=val_guides
)

# Option 2: Random split (leave lists empty)
datamodule = GenomicDataModule(
    df=df,
    test_guides=[],       # 20% of remaining data becomes test
    val_guides=[]         # If empty, 20% of non-test becomes val
)
```

### DataLoader

```python
# Training dataloader (with class-weighted oversampling)
train_loader = datamodule.train_dataloader()

# Validation dataloader
val_loader = datamodule.val_dataloader()

# Test dataloader
test_loader = datamodule.test_dataloader()

# Iterate through batches
for target_x, off_target_x, epi, y, counts, strand in train_loader:
    # target_x: [batch, seq_len] - Target with CLS token
    # off_target_x: [batch, window_size] - Off-target sequence
    # epi: [batch, num_epi, window_size] - Epigenetic features
    # y: [batch] - Labels
    # counts: [batch, 3] - Read counts (if Score_norm provided)
    # strand: [batch] - Strand indicator (0=plus, 1=minus)
    break
```

### Sample Iteration Example

```python
for batch in datamodule.train_dataloader():
    target_x, off_target_x, epi, y, counts, strand = batch
    print(f"Target shape: {target_x.shape}")        # [batch, 25] (24mer + CLS)
    print(f"Off-target shape: {off_target_x.shape}")  # [batch, 512]
    print(f"Epi shape: {epi.shape}")                 # [batch, num_epi, 512]
    print(f"Label shape: {y.shape}")                 # [batch]
    break
```

## Fine-tuning Pipeline

### Complete Fine-tuning Example

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, spearmanr

from CRISCross.models import CRISCross
from CRISCross.Datasets import GenomicDataModule

# ============ Configuration ============
DATA_CONFIG = {
    "fasta_path": "data/GRCh38.primary_assembly.genome.fa",
    "epi_features": ["ATAC"],
    "bw_dir": "AGTensors_Tcell",
    "batch_size": 32,
    "window_size": 512,
    "num_samples": 320,  # batch_size * 10
    "norm_epi": True,
}

MODEL_CONFIG = {
    "vocab_size": 5,
    "dropout": 0.2,
    "context_layers": 3,
    "hidden_dim": 512,
    "num_epi": 1,  # Set to len(DATA_CONFIG["epi_features"])
    "output_size": 1,
    "windowsize": 512,
    "merge": "early",
}

TRAIN_CONFIG = {
    "max_epochs": 10,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
}

# ============ Load Data ============
df = pd.read_csv("datasets/TCellDataset.tsv", sep="\t")
print(df.head())

# Split data (optional)
test_guides = df[df["split"] == "test"]["GuideID"].tolist() if "split" in df.columns else []
val_guides = df[df["split"] == "val"]["GuideID"].tolist() if "split" in df.columns else []

# Create data module
datamodule = GenomicDataModule(df=df, test_guides=test_guides, val_guides=val_guides, **DATA_CONFIG)

# ============ Setup Model ============
cfg = {**MODEL_CONFIG, "num_epi": len(DATA_CONFIG["epi_features"])}
model = CRISCross(**cfg)

# Load pre-trained weights
model.load_state_dict(torch.hub.load_state_dict_from_url(
    "https://huggingface.co/domonik/criscross-atac/resolve/main/model.pt"
))

# ============ Define Metrics Callback ============
class MetricsCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        all_preds = []
        all_labels = []

        for batch in trainer.val_dataloaders:
            target_x, off_target_x, epi, y, counts, strand = batch
            with torch.no_grad():
                logits, _ = pl_module(target_x, off_target_x, strand, epi)
            all_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            all_labels.extend(y.cpu().numpy())

        ap = average_precision_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0
        spearman = spearmanr(all_labels, all_preds).correlation

        pl_module.log("val_ap", ap)
        pl_module.log("val_auc", auc)
        pl_module.log("val_spearman", spearman)

# ============ Train ============
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="val_ap",
    filename="best-model-{epoch:02d}-{val_ap:.4f}",
    save_top_k=3,
    mode="max"
)

trainer = pl.Trainer(
    max_epochs=TRAIN_CONFIG["max_epochs"],
    accelerator="gpu",
    devices=1,
    precision="bf16-mixed",
    callbacks=[checkpoint_callback, MetricsCallback()]
)

model.train()
trainer.fit(model, datamodule)
```

## Inference

### Loading and Using the Fine-tuned Model

```python
import torch
from CRISCross.models import CRISCross

# Load model
model = CRISCross(**model_config)
model.load_state_dict(torch.load("fine_tuned_model.pt"))

# Single sample (23nt sequence)
target = torch.tensor([[0, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3]])  # 24 (23+CLS)
off_target = torch.randint(0, 5, (1, 512))
strand = torch.tensor([0])
epi = torch.zeros(1, 0, 512)

with torch.no_grad():
    logits, _ = model(target, off_target, strand, epi)
    prediction = torch.sigmoid(logits).item()
```

### Batch Inference

```python
def predict(model, dataloader, device="cuda"):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for target_x, off_target_x, epi, y, counts, strand in dataloader:
            target_x = target_x.to(device)
            off_target_x = off_target_x.to(device)
            epi = epi.to(device)
            strand = strand.to(device)

            logits, _ = model(target_x, off_target_x, strand, epi)
            predictions = torch.sigmoid(logits).cpu().numpy()

            all_predictions.extend(predictions.flatten())
            all_labels.extend(y.cpu().numpy())

    return all_predictions, all_labels
```

## API Reference

### CRISCross Model

```python
class CRISCross(nn.Module):
    def forward(
        self,
        x: torch.Tensor,          # [batch, seq_len] - Target sequence with CLS
        off_target_x: torch.Tensor,  # [batch, window_size] - Off-target sequence
        strand: torch.Tensor,     # [batch] - Strand (0 or 1)
        epi: Optional[torch.Tensor] = None  # [batch, num_epi, window_size]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cls_logits: [batch, output_size] - Prediction logits
            transformer_outputs: [batch, seq_len-1, hidden_dim] - Transformer features
        """
```

### GenomicDataModule

```python
class GenomicDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fasta_path: str,
        epi_features: List[str],
        bw_dir: Union[str, List[str]],
        window_size: int = 512,
        batch_size: int = 32,
        num_workers: int = 4,
        num_samples: int = 10000,
        norm_epi: bool = False,
        mode: str = "np",
        df: Optional[pd.DataFrame] = None,
        val_guides: Optional[List[str]] = None,
        test_guides: Optional[List[str]] = None
    ):
        """
        Args:
            fasta_path: Path to genome FASTA file
            epi_features: List of epigenetic features to use
            bw_dir: Directory with .npy epi feature files
            window_size: Off-target window size (default: 512)
            batch_size: Batch size for dataloaders
            num_workers: Number of dataloader workers
            num_samples: Number of samples for oversampling (if df provided)
            norm_epi: Auto-normalize epigenetic features
            mode: "np" for numpy memmaps, "bw" for bigwig (deprecated)
            df: If provided, uses FineTuningGenomicDataset for dataframe-based fine-tuning
            val_guides/test_guides: GuideIDs for split management
        """
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `batch_size` or `num_samples`
2. **Missing Epi Files**: Ensure `bw_dir` contains all required `.npy` files
3. **Chromosome Mismatch**: Ensure dataframe chromosome names match FASTA/epi file names

### Debug Tips

```python
# Check dataframe structure
print(df.columns)
print(f"Sequence lengths: {df['Guide_sequence'].apply(len).unique()}")
print(df["chr"].unique())

# Verify 23nt sequences
assert all(len(seq) == 23 for seq in df["Guide_sequence"]), "All sequences must be 23nt"

# Check datamodule setup
datamodule.setup()
print(f"Train set: {len(datamodule.train_dataset)}")
print(f"Val set: {len(datamodule.val_dataset)}")
print(f"Test set: {len(datamodule.test_dataset)}")
```

## License

This project is licensed under the same license as the original CRISCross repository.

## Citation

Please cite the original CRISCross paper when using this code in your research.
