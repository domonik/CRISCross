# CRISCross Models Documentation

This document describes the neural network architectures implemented in `models.py`.

## Overview

CRISCross implements a single primary model architecture for predicting CRISPR Cas9 off-target effects:

| Model | Type | Use Case |
|-------|------|----------|
| `CRISCross` | Transformer | Main architecture with cross-attention |

## Core Architecture: CRISCross

The `CRISCross` model is the primary architecture, combining transformer self-attention with cross-attention between target and off-target sequences.

### Input Encoding

```python
model = CRISCross(
    vocab_size=5,           # A, C, G, T, N (0 is padding)
    dropout=0.2,
    context_layers=3,       # Number of transformer layers
    hidden_dim=512,         # Model dimension
    num_epi=0,              # Number of epigenetic features
    output_size=1,          # Binary classification
    windowsize=512,         # Off-target sequence length
    merge="early"           # Early fusion with epigenetics
)
```

### Forward Pass

```python
# Inputs:
#   x: target sequence [batch, seq_len]
#   off_target_x: off-target sequence [batch, windowsize]
#   strand: strand indicator [batch]
#   epi: epigenetic features [batch, num_epi, windowsize] (optional)

logits, attention_weights = model(x, off_target_x, strand, epi)
```

### Architecture Details

1. **K-mer Convolution**: Target and off-target sequences are converted to k-mers (k=3) using positional encoding: `token_id = base_1 + vocab_size*base_2 + vocab_size^2*base_3`

2. **Embedding Layers**:
   - `target_embedding`: K-mer embeddings for target sequence
   - `ot_embedding`: K-mer embeddings for off-target sequence
   - `strand_embedding`: Adds strand information (0=plus, 1=minus)
   - `positional_encoding`: BERT-style learned absolute positional embeddings
   - `ot_positional_encoding`: Same for off-target (no type embedding)

3. **Transformer Blocks** (repeated `context_layers` times):
   - `self_attention`: Self-attention on target sequence
   - `self_ot_attention`: Self-attention on off-target sequence
   - `cross_attention1`: Target queries attend to centered off-target (23nt PAM region)
   - `cross_attention2`: Off-target queries attend to full target (bidirectional)

4. **Output**: CLS token representation projected to output dimension

### Key Components

#### HighlightCenterAndPAM

Adds token-type embeddings to highlight biologically significant regions:
- Center 23nt (PAM-proximal region): type=1
- Last 3nt (PAM sequence): type=2

```python
class HighlightCenterAndPAM(nn.Module):
    def forward(self, x, center):
        mask = torch.zeros_like(x[..., 0])
        mask[:, start:end] = 1      # center 23nt
        mask[:, end-3:end] = 2      # PAM
        return x + self.token_type_emb(mask.long())
```

#### LearnedPositionalEmbedding

BERT-style positional embeddings with type embeddings:
- Absolute positional IDs via `nn.Embedding`
- Type embedding for first position (CLS-like)
- LayerNorm and Dropout

#### SelfAttentionLayer

Standard transformer self-attention block:
```
Input -> LayerNorm -> MultiheadAttention -> Dropout -> Residual -> LayerNorm -> MLP -> Dropout -> Residual -> Output
```

#### CrossAttentionLayer

Cross-attention where query attends to separate key/value:
```
query -> LayerNorm -> CrossAttention(query, key_value, key_value) -> Dropout -> Residual
x -> LayerNorm -> MLP -> Dropout -> Residual -> Output
```

## Usage Example

```python
import torch
from CRISCross.models import CRISCross

# Initialize model
model = CRISCross(
    vocab_size=5,
    dropout=0.2,
    context_layers=3,
    hidden_dim=512,
    num_epi=1,
    output_size=1,
    windowsize=512,
    merge="early"
)

# Create batch
batch_size = 4
target_seq = torch.randint(0, 5, (batch_size, 25))  # Target with CLS
off_target = torch.randint(0, 5, (batch_size, 512))  # Off-target
strand = torch.randint(0, 2, (batch_size,))
epi = torch.randn(batch_size, 1, 512)  # One epigenetic feature

# Forward pass
logits, attn = model(target_seq, off_target, strand, epi)
print(logits.shape)  # [batch_size, 1]
```

## Training Considerations

1. **Sequence Encoding**: All sequences use k-mer (k=3) convolution for context
2. **Center Position**: Critical region is 23nt around off-target center (PAM-proximal)
3. **Strand Information**: Explicitly encoded via `StrandEmbedding`
4. **Epigenetic Fusion**: Early fusion adds epi features to off-target embeddings

## Architectural Decisions

- **Vocabulary Size**: 5 (A, C, G, T, N) - simplifies k-mer encoding to integer IDs
- **K-mer Size**: 3 - balances context length with vocabulary explosion (5^3 = 125)
- **Attention Heads**: 4 - fixed across all attention layers
- **MLP Ratio**: 4x hidden dimension in feed-forward layers
- **Dropout**: 0.2 - consistent regularization across models

## Helper Modules

### torch_convolve_int

K-mer conversion function that applies the k-mer kernel to convert token sequences to k-mer representations.

### torch_convolve_int

K-mer conversion function that applies the k-mer kernel to convert token sequences to k-mer representations.
