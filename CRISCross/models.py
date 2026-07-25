"""
CRISCross neural network models.

This module contains the CRISCross transformer-based architecture for
predicting CRISPR off-target effects, along with supporting components.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from typing import Optional, Tuple


def torch_convolve_int(tokens: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    1D convolution over integer-encoded token sequences using k-mer positional encoding.

    Converts sequences of token IDs to k-mer representations where each position is
    encoded as: base_1 + vocab_size * base_2 + vocab_size^2 * base_3 (for k=3).

    This enables the model to capture local context (k-mer windows) while maintaining
    the sequence structure. For k=3 with vocab_size=5, there are 5^3=125 possible
    k-mers, each mapped to a unique integer ID.

    Args:
        tokens: Input token sequence [batch, seq_len] with integer token IDs.
                Token IDs should be in range [0, vocab_size), where 0 typically
                represents padding or special tokens.
        kernel: K-mer kernel tensor [k] where kernel[i] = vocab_size^i. This kernel
                converts k-mer windows to unique integer IDs via positional encoding.

    Returns:
        K-mer encoded sequence [batch, seq_len - k + 1] where each element is a
        unique integer ID representing the k-mer at that position.

    Example:
        >>> tokens = torch.tensor([[1, 2, 3, 4]])  # batch=1, seq_len=4
        >>> kernel = torch.tensor([1, 5, 25])  # k=3, vocab_size=5
        >>> torch_convolve_int(tokens, kernel)
        tensor([[ 12,  37,  62]])  # k-mer IDs for positions (1,2,3) and (2,3,4)
    """
    k = len(kernel)
    pad_left = k // 2
    pad_right = k - 1 - pad_left
    tokens = F.pad(tokens, (pad_left, pad_right), mode="constant", value=0)
    windows = tokens.unfold(1, k, 1)
    return (windows * kernel).sum(-1)


class SelfAttentionLayer(nn.Module):
    """Transformer self-attention block with pre-LayerNorm residual connections.

    Implements a standard transformer block with pre-LayerNorm architecture:
    1. Self-attention with LayerNorm applied before attention, with residual connection
    2. Feed-forward MLP with LayerNorm applied before MLP, with residual connection

    The pre-LN design provides more stable training for deep networks by normalizing
    inputs before transformation, reducing gradient explosion risks.

    Example:
        >>> layer = SelfAttentionLayer(embed_dim=512, num_heads=8, dropout=0.1)
        >>> x = torch.randn(32, 100, 512)  # batch=32, seq_len=100, embed_dim=512
        >>> output = layer(x)
        >>> output.shape
        torch.Size([32, 100, 512])

    Args:
        embed_dim: Dimension of the embedding (d_model). Determines size of input/output
            tensors and internal attention projections.
        num_heads: Number of attention heads. Must divide embed_dim evenly. Default: 4.
            Multiple heads allow attending to different representation subspaces.
        dropout: Dropout probability for attention outputs and MLP layers. Controls
            regularization strength. Default: 0.1.
        mlp_ratio: Ratio for hidden dimension in feed-forward network. MLP expands to
            embed_dim * mlp_ratio then projects back. Default: 4 (e.g., 512 -> 2048 -> 512).

    Attributes:
        self_attn: Multi-head self-attention module.
        norm1: LayerNorm for attention residual connection.
        dropout1: Dropout for attention residual connection.
        mlp: Sequential feed-forward network (Linear -> ReLU -> Dropout -> Linear -> Dropout).
        norm2: LayerNorm for MLP residual connection.

    Input shape:
        x: [batch, seq_len, embed_dim]

    Output shape:
        [batch, seq_len, embed_dim]
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1, mlp_ratio: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through self-attention block.

        Computes self-attention where query, key, and value all come from input x.
        Applies pre-LayerNorm, attention with residual connection and dropout, then
        passes through MLP with another pre-LayerNorm and residual connection.

        Example:
            >>> layer = SelfAttentionLayer(embed_dim=512)
            >>> x = torch.randn(32, 100, 512)
            >>> mask = torch.triu(torch.ones(100, 100), diagonal=1).bool()
            >>> output = layer(x, attn_mask=mask)

        Args:
            x: Input tensor of shape [batch, seq_len, embed_dim]. Contains embeddings
                for each position in the sequence.
            attn_mask: Optional attention mask. Can be:
                - [seq_len, seq_len] for causal/masked attention
                - [batch, seq_len, seq_len] for batched masks
                Masks should have -inf for masked positions and 0 for unmasked.
            key_padding_mask: Optional boolean mask of shape [batch, seq_len] where
                True indicates padding tokens to ignore in attention computation.

        Returns:
            Output tensor of shape [batch, seq_len, embed_dim] containing transformed
            embeddings that incorporate information from all sequence positions via
            self-attention.
        """
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        x = self.norm2(x + self.mlp(x))
        return x


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer with pre-LayerNorm residual connections.

    Implements cross-attention where the query comes from one sequence (e.g., target)
    and key/value come from a different sequence (e.g., off-target). This enables
    bidirectional information flow between target and off-target sequences in the
    CRISCross architecture.

    The architecture mirrors SelfAttentionLayer but allows attention to be computed
    between different sequences, which is essential for modeling interactions between
    guide RNA and potential off-target binding sites.

    Example:
        >>> layer = CrossAttentionLayer(embed_dim=512, num_heads=8, dropout=0.1)
        >>> query = torch.randn(32, 100, 512)   # target sequence
        >>> key_value = torch.randn(32, 512, 512)  # off-target sequence
        >>> output = layer(query, key_value)
        >>> output.shape
        torch.Size([32, 100, 512])

    Args:
        embed_dim: Dimension of the embedding. Determines size of input/output tensors
            and internal attention projections.
        num_heads: Number of attention heads. Must divide embed_dim evenly. Default: 4.
            Multiple heads allow attending to different feature subspaces.
        dropout: Dropout probability for attention outputs and MLP layers. Controls
            regularization strength. Default: 0.1.
        mlp_ratio: Ratio for hidden dimension in feed-forward network. MLP expands to
            embed_dim * mlp_ratio then projects back. Default: 4.

    Attributes:
        cross_attn: Multi-head cross-attention module (query from one seq, kv from another).
        norm1: LayerNorm for cross-attention residual connection.
        dropout1: Dropout for cross-attention residual connection.
        mlp: Sequential feed-forward network (Linear -> ReLU -> Dropout -> Linear -> Dropout).
        norm2: LayerNorm for MLP residual connection.

    Input shapes:
        query: [batch, Lq, embed_dim] - Query sequence (e.g., target)
        key_value: [batch, Lkv, embed_dim] - Key/value sequence (e.g., off-target)

    Output shape:
        [batch, Lq, embed_dim] - Same shape as query input
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1, mlp_ratio: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through cross-attention layer.

        Computes cross-attention where query comes from one sequence and key/value
        come from another sequence. Applies pre-LayerNorm, cross-attention with
        residual connection and dropout, then passes through MLP with another
        pre-LayerNorm and residual connection.

        Example:
            >>> layer = CrossAttentionLayer(embed_dim=512)
            >>> query = torch.randn(32, 100, 512)
            >>> key_value = torch.randn(32, 512, 512)
            >>> output = layer(query, key_value)

        Args:
            query: Query tensor of shape [batch, Lq, embed_dim]. Comes from one
                sequence (e.g., target sequence embeddings).
            key_value: Key/value tensor of shape [batch, Lkv, embed_dim]. Comes from
                a different sequence (e.g., off-target sequence embeddings).
            attn_mask: Optional attention mask of shape [Lq, Lkv] or [batch, Lq, Lkv]
                where True indicates positions to attend to, or -inf for positions
                to mask.
            key_padding_mask: Optional boolean mask of shape [batch, Lkv] where
                True indicates padding tokens in key_value that should be ignored.

        Returns:
            Output tensor of shape [batch, Lq, embed_dim]. Contains transformed
            query embeddings that incorporate information from the key_value sequence
            via cross-attention.
        """
        attn_output, _ = self.cross_attn(query, key_value, key_value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.norm1(query + self.dropout1(attn_output))
        x = self.norm2(x + self.mlp(x))
        return x


class HighlightCenterAndPAM(nn.Module):
    """Adds token-type embeddings to highlight biologically significant regions.

    CRISPR Cas9 targeting has critical regions that deserve special attention from
    the model:
    - Center region (23nt around middle): PAM-proximal region where mismatches
      have strongest impact on binding affinity
    - PAM sequence (last 3nt): Protospacer Adjacent Motif required for Cas9 binding

    This module adds learnable token-type embeddings based on biological significance:
    - type=0: Background (no special embedding added)
    - type=1: Center 23nt PAM-proximal region
    - type=2: Last 3nt PAM sequence

    The model can learn to attend more carefully to these critical regions.

    Example:
        >>> module = HighlightCenterAndPAM(d_model=512, n_types=3)
        >>> x = torch.randn(32, 512, 512)  # batch=32, seq_len=512, d_model=512
        >>> output = module(x, center=256)
        >>> output.shape
        torch.Size([32, 512, 512])

    Args:
        d_model: Dimension of the model/embedding. Determines size of token-type
            embeddings. Should match the embedding dimension of input tensors.
        n_types: Number of distinct token types. Default: 3 (background=0, center=1,
            PAM=2). Must be at least 3 to cover all regions.

    Attributes:
        token_type_emb: Embedding table of shape [n_types, d_model] storing learnable
            type embeddings for each token type.

    Input shape:
        x: [batch, seq_len, d_model]

    Output shape:
        [batch, seq_len, d_model] - Input with type embeddings added
    """
    def __init__(self, d_model: int, n_types: int = 3):
        super().__init__()
        self.token_type_emb = nn.Embedding(n_types, d_model)

    def forward(self, x: torch.Tensor, center: int) -> torch.Tensor:
        """Apply token-type embeddings based on biological significance.

        Creates a mask marking:
        - Center 23nt region (PAM-proximal): type=1
        - Last 3nt (PAM sequence): type=2
        - Rest of sequence: type=0 (background, no embedding added)

        The center parameter determines the middle position from which the 23nt
        PAM-proximal region is centered. The PAM (last 3nt) is always at the end
        of the sequence.

        Example:
            >>> module = HighlightCenterAndPAM(d_model=512)
            >>> x = torch.randn(4, 512, 512)
            >>> output = module(x, center=256)  # Center at position 256
            # Positions 245-267 get type=1 (center), 509-512 get type=2 (PAM)

        Args:
            x: Input tensor of shape [batch, seq_len, d_model]. Embeddings to which
                type embeddings will be added.
            center: Center index for the 23nt PAM-proximal region. The region spans
                from (center - 12) to (center + 11), covering 23 positions centered
                around this index.

        Returns:
            Tensor of shape [batch, seq_len, d_model] with token-type embeddings
            added. Each position has an embedding corresponding to its biological
            significance (background, PAM-proximal, or PAM sequence).
        """
        mask = torch.zeros(x.size(0), x.size(1), device=x.device)
        start = center - 23//2 - 1
        end = center + 23//2
        mask[:, start:end] = 1
        mask[:, end-3:end] = 2
        return x + self.token_type_emb(mask.long())


class LearnedPositionalEmbedding(nn.Module):
    """BERT-style learned absolute positional embeddings with optional type embedding.

    Uses learnable embedding table for positions rather than fixed sinusoidal
    encodings (like Transformer). This approach, used in BERT, allows the model
    to learn optimal positional representations from data.

    Additionally adds a special type embedding to the first position (index 0),
    creating a CLS-like token that can serve as a sequence-level representation.
    This is useful for classification tasks where a single token should aggregate
    information from the entire sequence.

    The module applies LayerNorm after adding positional embeddings for stable
    training, followed by dropout for regularization.

    Example:
        >>> pos_embed = LearnedPositionalEmbedding(
        ...     max_position_embeddings=512,
        ...     hidden_size=512,
        ...     dropout=0.1
        ... )
        >>> x = torch.randn(32, 100, 512)  # batch=32, seq_len=100, embed_dim=512
        >>> output = pos_embed(x)
        >>> output.shape
        torch.Size([32, 100, 512])

    Args:
        max_position_embeddings: Maximum sequence length supported. Determines size
            of position embeddings table. Position IDs from 0 to max_position_embeddings-1
            are valid.
        hidden_size: Dimension of the embeddings. Must match the embedding dimension
            of input tensors.
        dropout: Dropout probability applied after positional encoding and LayerNorm.
            Controls regularization strength.
        add_type: Whether to add type embedding for first position. When True, the
            first position (index 0) receives an additional learnable type embedding,
            creating a CLS-like token. Default: True.

    Attributes:
        position_embeddings: Learnable embedding table of shape
            [max_position_embeddings, hidden_size].
        type_embedding: Learnable embedding table of shape [2, hidden_size] for
            type embedding at first position (when add_type=True).
        dropout: Dropout layer applied after positional encoding.
        norm: LayerNorm applied after positional encoding for stable training.

    Input shape:
        x: [batch, seq_len, hidden_size] or [batch, seq_len] - can accept embedded
            tensors or raw position IDs

    Output shape:
        [batch, seq_len, hidden_size] - Positionally encoded embeddings
    """
    def __init__(
        self,
        max_position_embeddings: int,
        hidden_size: int,
        dropout: float,
        add_type: bool = True,
    ):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.type_embedding = nn.Embedding(2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.add_type = add_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learned positional embeddings to input tensor.

        Computes absolute position IDs from input shape and looks up corresponding
        positional embeddings. Adds them to input (if embedded) or replaces input
        (if raw position IDs). Optionally adds type embedding to first position.
        Applies dropout and LayerNorm for stable training.

        Example:
            >>> pos_embed = LearnedPositionalEmbedding(max_position_embeddings=512, hidden_size=512, dropout=0.1)
            >>> x = torch.randn(32, 100, 512)
            >>> output = pos_embed(x)  # Adds position 0-99 embeddings to x

        Args:
            x: Input tensor. Can be:
                - [batch, seq_len, hidden_size]: Embedded tensor where positional
                  embeddings will be added.
                - [batch, seq_len]: Raw position IDs that will be converted to
                  embeddings.

        Returns:
            Tensor of shape [batch, seq_len, hidden_size] with positional embeddings
            added. Contains input embeddings with absolute position information
            encoded, plus type embedding at first position if add_type=True.
        """
        B, L = x.shape[:2]
        position_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(x.size(0), -1)

        res = self.position_embeddings(position_ids) + x
        if self.add_type:
            token_type_ids = torch.zeros(B, L, dtype=torch.long, device=x.device)
            token_type_ids[:, 0] = 1
            res += self.type_embedding(token_type_ids)
        return self.dropout(self.norm(res))


class StrandEmbedding(nn.Module):
    """Adds strand-specific embeddings to sequence representations.

    CRISPR Cas9 targeting is strand-specific - the guide RNA binds to the
    complementary strand and the PAM must appear on the non-target strand.
    This module adds learnable strand embeddings to inform the model which
    DNA strand is being processed.

    The strand embedding is broadcast and added to all positions in the sequence,
    allowing the model to condition its predictions on strand orientation.

    Example:
        >>> strand_emb = StrandEmbedding(d_model=512)
        >>> x = torch.randn(32, 100, 512)  # batch=32, seq_len=100, d_model=512
        >>> strand_ids = torch.tensor([0, 1, 0, 1])  # batch=4, 0=plus, 1=minus
        >>> output = strand_emb(x, strand_ids)
        >>> output.shape
        torch.Size([32, 100, 512])

    Args:
        d_model: Dimension of the model/embedding. Determines size of strand
            embeddings. Should match the embedding dimension of input tensors.

    Attributes:
        strand_emb: Embedding table of shape [2, d_model] storing learnable
            embeddings for plus strand (0) and minus strand (1).

    Input shapes:
        x: [batch, seq_len, d_model] - Input embeddings to which strand
            information will be added.
        strand_ids: [batch, seq_len] or [batch] - Strand indicator where
            0 represents plus strand and 1 represents minus strand.

    Output shape:
        [batch, seq_len, d_model] - Input with strand embeddings added
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.strand_emb = nn.Embedding(2, d_model)

    def forward(self, x: torch.Tensor, strand_ids: torch.Tensor) -> torch.Tensor:
        """Add strand-specific embeddings to input tensor.

        Looks up strand embeddings based on strand_ids and adds them to input.
        The strand embedding is broadcast across all sequence positions, allowing
        the model to condition its processing on strand orientation.

        Example:
            >>> strand_emb = StrandEmbedding(d_model=512)
            >>> x = torch.randn(4, 100, 512)
            >>> strand_ids = torch.tensor([0, 1, 0, 1])  # 2 plus, 2 minus
            >>> output = strand_emb(x, strand_ids)
            # Positions for batch 0,2 get plus strand emb; batch 1,3 get minus

        Args:
            x: Input tensor of shape [batch, seq_len, d_model]. Embeddings to
                which strand information will be added.
            strand_ids: Strand indicator tensor of shape [batch] or [batch, seq_len].
                Values should be 0 (plus strand) or 1 (minus strand). If shape is
                [batch, seq_len], the same strand is applied to all positions for
                each batch element.

        Returns:
            Tensor of shape [batch, seq_len, d_model] with strand embeddings added.
            Each batch element has strand-specific information encoded based on
            whether it represents plus or minus strand DNA.
        """
        return x + self.strand_emb(strand_ids).unsqueeze(1)


class CRISCross(nn.Module):
    """Transformer-based model for predicting CRISPR Cas9 off-target effects.

    CRISCross combines self-attention on target and off-target sequences with
    bidirectional cross-attention to model interactions between guide RNA and
    potential off-target binding sites. The architecture is designed to capture
    both local sequence context (via k-mer encoding) and long-range dependencies
    (via self-attention).

    Key design decisions:
    - K-mer (k=3) convolution captures local 3-nucleotide context while reducing
      sequence length and vocabulary complexity
    - Bidirectional cross-attention allows target to attend to PAM-proximal
      off-target region and vice versa
    - PAM highlighting (token-type embeddings) emphasizes biologically critical
      regions: center 23nt (mismatch-sensitive) and last 3nt (PAM sequence)
    - Optional epigenetic feature fusion enables context-specific predictions

    Architecture pipeline:
    1. K-mer conversion: Target and off-target sequences converted to k-mers (k=3)
    2. K-mer embeddings: Separate embedding tables for target and off-target k-mers
    3. Positional encoding: BERT-style learned absolute positional embeddings
    4. PAM highlighting: Token-type embeddings for center and PAM regions
    5. Strand embedding: Adds strand orientation information to off-target
    6. Epigenetic fusion (optional): Early fusion adds epi features to off-target
    7. Transformer layers: Alternating self-attention and bidirectional cross-attention
    8. Output projection: CLS token projected to prediction space

    Example:
        >>> model = CRISCross(
        ...     vocab_size=5, dropout=0.2, context_layers=3,
        ...     hidden_dim=512, num_epi=1, output_size=1,
        ...     windowsize=512, merge="early"
        ... )
        >>> batch_size = 4
        >>> target = torch.randint(0, 5, (batch_size, 25))   # Target with CLS
        >>> off_target = torch.randint(0, 5, (batch_size, 512))
        >>> strand = torch.randint(0, 2, (batch_size,))
        >>> epi = torch.randn(batch_size, 1, 512)
        >>> logits, attn = model(target, off_target, strand, epi)
        >>> logits.shape
        torch.Size([4, 1])

    Args:
        vocab_size: Vocabulary size for DNA bases. Default: 5 (A=1, C=2, G=3, T=4, N=0).
            Token ID 0 is used for padding.
        dropout: Dropout probability applied throughout the model. Controls
            regularization strength. Default: 0.2.
        context_layers: Number of transformer layers (self-attention + cross-attention
            blocks). More layers capture longer-range dependencies but increase
            computation. Default: 3.
        hidden_dim: Hidden dimension size (d_model). Determines embedding sizes and
            internal representation dimensionality. Default: 512.
        num_epi: Number of epigenetic features. Set to 0 for sequence-only model.
            When > 0, early fusion adds epigenetic features to off-target embeddings.
        output_size: Output dimension. Typically 1 for binary classification (off-target
            probability) or regression (binding energy).
        windowsize: Off-target sequence length. This is the window size around the
            PAM where off-target sites are evaluated. Default: 512.
        merge: Fusion strategy for epigenetic features. Options:
            - "early": Add epi features to off-target embeddings (requires num_epi > 0)
            - None: Ignore epigenetic features entirely
            - Other values raise error (future fusion strategies)

    Attributes:
        kernel: K-mer conversion kernel buffer of shape [3] where kernel[i] =
            vocab_size^i. Used to convert k-mer windows to unique integer IDs.
        m_k: Number of possible k-mers (vocab_size^3). Size of k-mer embedding tables.
        target_embedding: Embedding table for target k-mers of shape [m_k, hidden_dim].
        ot_embedding: Embedding table for off-target k-mers of shape [m_k, hidden_dim].
        self_attention: ModuleList of target self-attention layers.
        self_ot_attention: ModuleList of off-target self-attention layers.
        cross_attention1: ModuleList for target attending to off-target PAM region.
        cross_attention2: ModuleList for off-target PAM region attending to target.
        out_proj: Output projection head (Linear -> Tanh -> Dropout -> Linear).

    Input shapes:
        x: Target sequence [batch, seq_len]. Should include CLS token at position
            0, so typically [batch, 1 + guide_length] where guide_length is ~20.
        off_target_x: Off-target sequence [batch, windowsize]. Full window around
            PAM position.
        strand: Strand indicator [batch]. Values: 0 (plus strand) or 1 (minus strand).
        epi: Epigenetic features [batch, num_epi, windowsize]. Optional feature
            matrix where each channel is a different epigenetic mark (ATAC, histone
            marks, etc.). Required when merge="early" and num_epi > 0.

    Returns:
        Tuple of (cls_logits, transformer_outputs):
        - cls_logits: [batch, output_size] - Final prediction logits. For
            output_size=1, represents off-target probability (via sigmoid) or
            binding energy.
        - transformer_outputs: [batch, seq_len - 1, hidden_dim] - Transformer
            representations for target positions (excluding CLS token). Can be
            used for attention analysis or as features for downstream tasks.
    """
    def __init__(
        self,
        vocab_size: int,
        dropout: float,
        context_layers: int,
        hidden_dim: int,
        num_epi: int,
        output_size: int,
        windowsize: int,
        merge: str,
    ):
        super().__init__()
        self.merge = merge
        self.kernel_size = 3
        self.transformer_dim = hidden_dim
        self.dropout = dropout
        self.vocab_size = vocab_size

        # K-mer conversion kernel: vocab_size^0, vocab_size^1, vocab_size^2
        self.register_buffer(
            "kernel",
            torch.tensor([self.vocab_size ** i for i in range(self.kernel_size)], dtype=torch.long)
        )
        self.m_k = self.vocab_size ** self.kernel_size

        # Token type embeddings for PAM highlighting
        self.token_type_emb = HighlightCenterAndPAM(hidden_dim, 3)

        # K-mer embeddings
        self.target_embedding = nn.Embedding(self.m_k, hidden_dim)
        self.ot_embedding = nn.Embedding(self.m_k, hidden_dim)

        # Strand and positional encoding
        self.strand_embedding = StrandEmbedding(hidden_dim)
        self.positional_encoding = LearnedPositionalEmbedding(
            hidden_size=hidden_dim,
            max_position_embeddings=32,
            dropout=0.1
        )
        self.ot_positional_encoding = LearnedPositionalEmbedding(
            hidden_size=hidden_dim,
            max_position_embeddings=windowsize,
            dropout=0.1,
            add_type=False
        )

        # Epigenetic feature embedding (for early fusion)
        self.epi_embeddor = nn.Sequential(
            nn.Linear(num_epi, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.ndrop = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Transformer layers
        self.n_layers = context_layers
        self.self_attention = nn.ModuleList(
            SelfAttentionLayer(embed_dim=hidden_dim, dropout=dropout, mlp_ratio=4, num_heads=4)
            for _ in range(self.n_layers)
        )
        self.self_ot_attention = nn.ModuleList(
            SelfAttentionLayer(embed_dim=hidden_dim, dropout=dropout, mlp_ratio=4, num_heads=4)
            for _ in range(self.n_layers)
        )
        self.cross_attention1 = nn.ModuleList(
            CrossAttentionLayer(embed_dim=hidden_dim, dropout=dropout, mlp_ratio=4)
            for _ in range(self.n_layers)
        )
        self.cross_attention2 = nn.ModuleList(
            CrossAttentionLayer(embed_dim=hidden_dim, dropout=dropout, mlp_ratio=4)
            for _ in range(self.n_layers)
        )

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features=output_size)
        )

    def forward(
        self,
        x: torch.Tensor,
        off_target_x: torch.Tensor,
        strand: torch.Tensor,
        epi: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through CRISCross model.

        Processes target and off-target sequences through the transformer architecture:
        1. Converts sequences to k-mers and adds CLS token to target
        2. Applies k-mer embeddings and positional encodings
        3. Highlights PAM-proximal and PAM regions with token-type embeddings
        4. Adds strand information to off-target embeddings
        5. Optionally fuses epigenetic features (early fusion)
        6. Processes through transformer layers (self-attention + cross-attention)
        7. Projects CLS token to output space

        The cross-attention pattern is asymmetric:
        - Target attends to centered off-target region (23nt around PAM)
        - Off-target center attends back to full target (bidirectional)

        Example:
            >>> model = CRISCross(
            ...     vocab_size=5, dropout=0.2, context_layers=3,
            ...     hidden_dim=512, num_epi=1, output_size=1,
            ...     windowsize=512, merge="early"
            ... )
            >>> target = torch.randint(0, 5, (4, 25))
            >>> off_target = torch.randint(0, 5, (4, 512))
            >>> strand = torch.tensor([0, 1, 0, 1])
            >>> epi = torch.randn(4, 1, 512)
            >>> logits, outputs = model(target, off_target, strand, epi)
            >>> logits.shape
            torch.Size([4, 1])

        Args:
            x: Target sequence tensor of shape [batch, seq_len]. Contains guide
                RNA sequence with CLS token prepended. Token IDs should be in
                range [0, vocab_size) where 0 is padding.
            off_target_x: Off-target sequence tensor of shape [batch, windowsize].
                Contains potential off-target binding site around PAM position.
                Token IDs in range [0, vocab_size).
            strand: Strand indicator tensor of shape [batch]. Values should be
                0 (plus strand) or 1 (minus strand). Determines which DNA strand
                is the target for Cas9 binding.
            epi: Optional epigenetic features tensor of shape [batch, num_epi,
                windowsize]. Each of the num_epi channels represents a different
                epigenetic mark (ATAC-seq, ChIP-seq, etc.). Only used when
                merge="early" and num_epi > 0.

        Returns:
            Tuple of (cls_logits, transformer_outputs):
            - cls_logits: Tensor of shape [batch, output_size] containing final
                prediction logits. For output_size=1, typically represents
                off-target probability or binding energy.
            - transformer_outputs: Tensor of shape [batch, seq_len - 1, hidden_dim]
                containing transformer representations for target positions
                (excluding CLS token). Can be used for attention visualization
                or as features for auxiliary tasks.
        """
        strand = strand.long()
        x = x.long()
        off_target_x = off_target_x.long()

        # Convert to k-mers
        x = torch_convolve_int(x, self.kernel)
        cls = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        x = torch.cat([cls, x], dim=1)

        ot = torch_convolve_int(off_target_x, self.kernel)
        center = ot.shape[1] // 2 + ot.shape[1] % 2

        # Embeddings
        x = self.target_embedding(x)
        x = self.positional_encoding(x)

        ot = self.ot_embedding(ot)
        ot = self.token_type_emb(ot, center)
        ot = self.strand_embedding(ot, strand)
        ot = self.ot_positional_encoding(ot)

        # Early fusion with epigenetic features
        if self.merge == "early" and epi is not None:
            epi = self.epi_embeddor(epi)
            ot = ot + epi
            ot = self.ndrop(ot)

        # Transformer layers
        for i in range(self.n_layers):
            x = self.self_attention[i](x)
            ot = self.self_ot_attention[i](ot)
            x_old = x
            x = self.cross_attention1[i](x, ot[:, center-23//2-1:center+23//2])
            if i < self.n_layers - 1:
                ot[:, center-23//2-1:center+23//2] = self.cross_attention2[i](
                    ot[:, center-23//2-1:center+23//2], x_old
                )

        return self.out_proj(x[:, 0]), x[:, 1:]


def get_lora_config(r: int = 8, alpha: int = 16, dropout: float = 0.1):
    """Return a LoraConfig targeting the linear layers of CRISCross attention and MLP blocks.

    nn.MultiheadAttention stores the q/k/v projections as a fused in_proj_weight parameter
    (not as nn.Linear submodules), so PEFT cannot wrap them directly. We target the next
    best set: the attention output projection (out_proj) and both MLP linear layers (mlp.0,
    mlp.3) in every SelfAttentionLayer and CrossAttentionLayer.

    Args:
        r: LoRA rank. Lower values mean fewer trainable parameters. Default: 8.
        alpha: LoRA scaling factor (effective scale = alpha / r). Default: 16.
        dropout: Dropout applied to the LoRA adapter path. Default: 0.1.

    Returns:
        peft.LoraConfig configured for CRISCross.
    """
    from peft import LoraConfig
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        # Use fully-qualified suffixes to avoid matching CRISCross.out_proj (a Sequential).
        # "out_proj" alone would match that Sequential and crash; these only match the
        # nn.Linear out_proj inside nn.MultiheadAttention (self_attn / cross_attn).
        target_modules=["self_attn.out_proj", "cross_attn.out_proj", "mlp.0", "mlp.3"],
        lora_dropout=dropout,
        bias="none",
    )


def freeze_base_model(model: nn.Module) -> None:
    """Freeze all parameters that are not part of a LoRA adapter.

    After calling get_peft_model(), this freezes every parameter whose name does
    not contain "lora", leaving only the LoRA adapter weights trainable.

    Args:
        model: The PEFT-wrapped model (returned by get_peft_model).
    """
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all model parameters (for full fine-tuning).

    Args:
        model: Any nn.Module.
    """
    for param in model.parameters():
        param.requires_grad = True


