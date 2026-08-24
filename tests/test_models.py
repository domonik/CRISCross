"""
Pytest tests for CRISCross/models.py

Tests cover:
- Utility functions (torch_convolve_int)
- Embedding modules (HighlightCenterAndPAM, LearnedPositionalEmbedding, StrandEmbedding)
- Attention layers (SelfAttentionLayer, CrossAttentionLayer)
- Main CRISCross model
"""

import pytest
import torch
import torch.nn as nn
from CRISCross.models import (
    torch_convolve_int,
    SelfAttentionLayer,
    CrossAttentionLayer,
    HighlightCenterAndPAM,
    LearnedPositionalEmbedding,
    StrandEmbedding,
    CRISCross,
)
from CRISCross.bulges import band_bounds, band_width


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def cpu_device():
    """Ensure tests run on CPU."""
    return "cpu"


@pytest.fixture
def seed():
    """Random seed for reproducibility."""
    return 42


# ============================================================================
# Tests for torch_convolve_int
# ============================================================================

class TestTorchConvolveInt:
    """Tests for the k-mer conversion utility function."""

    def test_basic_kmer_conversion(self, cpu_device):
        """Test basic k-mer conversion with known input."""
        # Input: batch=1, seq_len=4
        tokens = torch.tensor([[1, 2, 3, 4]], device=cpu_device)
        kernel = torch.tensor([1, 5, 25], device=cpu_device)  # k=3, vocab_size=5

        result = torch_convolve_int(tokens, kernel)

        # Function preserves sequence length via padding
        # Position 0: (0 + 1*5 + 2*25) = 55 (with zero padding)
        # Position 1: (1 + 2*5 + 3*25) = 86
        # Position 2: (2 + 3*5 + 4*25) = 117
        # Position 3: (3 + 4*5 + 0*25) = 23 (with zero padding)
        expected = torch.tensor([[55, 86, 117, 23]], device=cpu_device)
        assert torch.equal(result, expected)

    def test_sequence_length_reduction(self, cpu_device):
        """Test that output sequence preserves length due to padding."""
        tokens = torch.randint(0, 5, (2, 10), device=cpu_device)
        kernel = torch.tensor([1, 5, 25], device=cpu_device)

        result = torch_convolve_int(tokens, kernel)

        # Function preserves sequence length via padding
        assert result.shape == (2, 10)

    def test_padding_behavior(self, cpu_device):
        """Test that padding is applied at sequence boundaries."""
        # Sequence with zeros at start (should be treated as padding)
        tokens = torch.tensor([[0, 0, 1, 2, 3]], device=cpu_device)
        kernel = torch.tensor([1, 5, 25], device=cpu_device)

        result = torch_convolve_int(tokens, kernel)

        # Output preserves sequence length due to padding
        assert result.shape == (1, 5)

    def test_batch_processing(self, cpu_device):
        """Test batch processing with multiple sequences."""
        tokens = torch.tensor([
            [1, 2, 3, 4, 5],
            [4, 3, 2, 1, 0],
        ], device=cpu_device)
        kernel = torch.tensor([1, 5, 25], device=cpu_device)

        result = torch_convolve_int(tokens, kernel)

        # Output preserves sequence length
        assert result.shape == (2, 5)
        # Each batch element should be processed independently
        assert not torch.equal(result[0, :], result[1, :])  # Different inputs -> different outputs

    def test_single_kmer(self, cpu_device):
        """Test with k=1 (single nucleotide)."""
        tokens = torch.tensor([[1, 2, 3, 4]], device=cpu_device)
        kernel = torch.tensor([1], device=cpu_device)

        result = torch_convolve_int(tokens, kernel)

        # k=1, no sequence reduction
        assert result.shape == (1, 4)
        assert torch.equal(result, tokens)


# ============================================================================
# Tests for HighlightCenterAndPAM
# ============================================================================

class TestHighlightCenterAndPAM:
    """Tests for biological region highlighting embeddings."""

    def test_token_type_embeddings_added(self, cpu_device):
        """Test that token type embeddings are added to input."""
        d_model = 64
        module = HighlightCenterAndPAM(d_model=d_model, n_types=3)
        module = module.to(cpu_device)

        batch_size = 2
        seq_len = 512
        x = torch.randn(batch_size, seq_len, d_model, device=cpu_device)
        start, end = band_bounds(seq_len)

        output = module(x, start, end)

        # Output should have same shape as input
        assert output.shape == x.shape
        # Output should be different from input (embeddings added)
        assert not torch.equal(output, x)

    def test_correct_region_masking(self, cpu_device):
        """Test that the band and its trailing 3nt PAM are correctly marked."""
        d_model = 32
        module = HighlightCenterAndPAM(d_model=d_model, n_types=3)
        module = module.to(cpu_device)

        batch_size = 1
        seq_len = 512
        x = torch.zeros(batch_size, seq_len, d_model, device=cpu_device)
        start, end = band_bounds(seq_len)

        output = module(x, start, end)

        band = module.token_type_emb.weight[1]
        pam = module.token_type_emb.weight[2]
        background = module.token_type_emb.weight[0]

        assert torch.allclose(output[0, start], band)
        assert torch.allclose(output[0, end - 4], band)
        assert torch.allclose(output[0, end - 3], pam)
        assert torch.allclose(output[0, end - 1], pam)
        assert torch.allclose(output[0, end], background)
        assert torch.allclose(output[0, start - 1], background)

    def test_region_label_is_fixed_width(self, cpu_device):
        """The labelled extent must not depend on anything sample-specific.

        A region label that tracked the true 19/20/21-nt protospacer extent
        would hand the model the bulge configuration (spec section 6.2).
        """
        d_model = 16
        module = HighlightCenterAndPAM(d_model=d_model, n_types=3).to(cpu_device)
        seq_len = 512

        for delta in [0, 1, 2, 3]:
            start, end = band_bounds(seq_len, delta)
            x = torch.zeros(1, seq_len, d_model, device=cpu_device)
            output = module(x, start, end)
            labelled = (
                ~torch.isclose(output[0], module.token_type_emb.weight[0]).all(-1)
            ).sum()
            assert int(labelled) == band_width(delta)

    def test_multiple_batch_elements(self, cpu_device):
        """Test with multiple batch elements."""
        d_model = 64
        module = HighlightCenterAndPAM(d_model=d_model, n_types=3)
        module = module.to(cpu_device)

        batch_size = 4
        seq_len = 256
        x = torch.randn(batch_size, seq_len, d_model, device=cpu_device)
        start, end = band_bounds(seq_len)

        output = module(x, start, end)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_different_band_positions(self, cpu_device):
        """Test with different band placements."""
        d_model = 32
        module = HighlightCenterAndPAM(d_model=d_model, n_types=3)
        module = module.to(cpu_device)

        x = torch.randn(1, 100, d_model, device=cpu_device)

        for start in [10, 40, 70]:
            output = module(x.clone(), start, start + 23)
            assert output.shape == x.shape


# ============================================================================
# Tests for LearnedPositionalEmbedding
# ============================================================================

class TestLearnedPositionalEmbedding:
    """Tests for BERT-style positional embeddings."""

    def test_positional_encoding_applied(self, cpu_device):
        """Test that positional encoding is applied to input."""
        max_pos = 128
        hidden_size = 64
        module = LearnedPositionalEmbedding(
            max_position_embeddings=max_pos,
            hidden_size=hidden_size,
            dropout=0.0  # Disable dropout for testing
        )
        module = module.to(cpu_device)

        batch_size = 2
        seq_len = 50
        x = torch.randn(batch_size, seq_len, hidden_size, device=cpu_device)

        output = module(x)

        assert output.shape == x.shape
        # Output should be different from input
        assert not torch.equal(output, x)

    def test_cls_type_embedding_at_position_0(self, cpu_device):
        """Test that first position gets type embedding."""
        max_pos = 128
        hidden_size = 32
        module = LearnedPositionalEmbedding(
            max_position_embeddings=max_pos,
            hidden_size=hidden_size,
            dropout=0.0,
            add_type=True
        )
        module = module.to(cpu_device)

        x = torch.zeros(1, 10, hidden_size, device=cpu_device)
        output = module(x)

        # First position should have type embedding added
        # Compare with module without type embedding
        module_no_type = LearnedPositionalEmbedding(
            max_position_embeddings=max_pos,
            hidden_size=hidden_size,
            dropout=0.0,
            add_type=False
        ).to(cpu_device)

        output_no_type = module_no_type(x)

        # First position should be different
        assert not torch.allclose(output[:, 0, :], output_no_type[:, 0, :], atol=1e-5)

    def test_layer_norm_applied(self, cpu_device):
        """Test that LayerNorm is applied."""
        max_pos = 128
        hidden_size = 64
        module = LearnedPositionalEmbedding(
            max_position_embeddings=max_pos,
            hidden_size=hidden_size,
            dropout=0.0
        )
        module = module.to(cpu_device)

        x = torch.randn(2, 50, hidden_size, device=cpu_device)
        output = module(x)

        # LayerNorm preserves mean ~0 and std ~1 (approximately)
        # This is a soft check due to positional embedding addition
        assert output.std() > 0

    def test_dropout_in_train_mode(self, cpu_device, seed):
        """Test that dropout is applied in training mode."""
        torch.manual_seed(seed)
        max_pos = 128
        hidden_size = 64
        module = LearnedPositionalEmbedding(
            max_position_embeddings=max_pos,
            hidden_size=hidden_size,
            dropout=0.5
        )
        module = module.to(cpu_device)
        module.train()

        x = torch.ones(1, 10, hidden_size, device=cpu_device)
        output1 = module(x)
        output2 = module(x)

        # With dropout, outputs should differ
        assert not torch.allclose(output1, output2)

    def test_dropout_disabled_in_eval_mode(self, cpu_device, seed):
        """Test that dropout is disabled in evaluation mode."""
        torch.manual_seed(seed)
        max_pos = 128
        hidden_size = 64
        module = LearnedPositionalEmbedding(
            max_position_embeddings=max_pos,
            hidden_size=hidden_size,
            dropout=0.5
        )
        module = module.to(cpu_device)
        module.eval()

        x = torch.ones(1, 10, hidden_size, device=cpu_device)
        output1 = module(x)
        output2 = module(x)

        # Without dropout, outputs should be identical
        assert torch.allclose(output1, output2)


# ============================================================================
# Tests for StrandEmbedding
# ============================================================================

class TestStrandEmbedding:
    """Tests for strand-specific embeddings."""

    def test_strand_embeddings_added(self, cpu_device):
        """Test that strand embeddings are added to input."""
        d_model = 64
        module = StrandEmbedding(d_model=d_model)
        module = module.to(cpu_device)

        batch_size = 2
        seq_len = 100
        x = torch.randn(batch_size, seq_len, d_model, device=cpu_device)
        strand_ids = torch.tensor([0, 1], device=cpu_device)

        output = module(x, strand_ids)

        assert output.shape == x.shape
        assert not torch.equal(output, x)

    def test_plus_minus_strand_different(self, cpu_device):
        """Test that plus and minus strands get different embeddings."""
        d_model = 32
        module = StrandEmbedding(d_model=d_model)
        module = module.to(cpu_device)

        x = torch.zeros(2, 50, d_model, device=cpu_device)
        strand_ids = torch.tensor([0, 1], device=cpu_device)

        output = module(x, strand_ids)

        # Plus and minus strand should have different embeddings
        assert not torch.allclose(output[0, :, :], output[1, :, :], atol=1e-5)

    def test_broadcast_across_positions(self, cpu_device):
        """Test that strand embedding is broadcast across sequence positions."""
        d_model = 32
        module = StrandEmbedding(d_model=d_model)
        module = module.to(cpu_device)

        x = torch.zeros(1, 100, d_model, device=cpu_device)
        strand_ids = torch.tensor([0], device=cpu_device)

        output = module(x, strand_ids)

        # All positions should have the same strand embedding added
        for i in range(1, 100):
            assert torch.allclose(output[:, 0, :], output[:, i, :], atol=1e-5)

    def test_input_shape_batch_only(self, cpu_device):
        """Test with [batch] strand shape."""
        d_model = 64
        module = StrandEmbedding(d_model=d_model)
        module = module.to(cpu_device)

        x = torch.randn(4, 50, d_model, device=cpu_device)
        strand_ids = torch.tensor([0, 1, 0, 1], device=cpu_device)

        output = module(x, strand_ids)
        assert output.shape == x.shape

    def test_input_shape_batch_seq(self, cpu_device):
        """Test with [batch, 1] strand shape (broadcast across positions)."""
        d_model = 64
        module = StrandEmbedding(d_model=d_model)
        module = module.to(cpu_device)

        x = torch.randn(4, 50, d_model, device=cpu_device)
        # StrandEmbedding expects [batch] or [batch, 1] shape for broadcasting
        strand_ids = torch.tensor([0, 1, 0, 1], device=cpu_device)  # [batch] shape

        output = module(x, strand_ids)
        assert output.shape == x.shape


# ============================================================================
# Tests for SelfAttentionLayer
# ============================================================================

class TestSelfAttentionLayer:
    """Tests for transformer self-attention layer."""

    def test_output_shape_preserved(self, cpu_device):
        """Test that output shape matches input shape."""
        embed_dim = 128
        num_heads = 4
        layer = SelfAttentionLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0)
        layer = layer.to(cpu_device)

        batch_size = 2
        seq_len = 64
        x = torch.randn(batch_size, seq_len, embed_dim, device=cpu_device)

        output = layer(x)

        assert output.shape == x.shape

    def test_gradient_flow(self, cpu_device):
        """Test that gradients flow through the layer."""
        embed_dim = 64
        layer = SelfAttentionLayer(embed_dim=embed_dim, dropout=0.0)
        layer = layer.to(cpu_device)
        layer.train()

        x = torch.randn(2, 32, embed_dim, device=cpu_device, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_dropout_in_train_mode(self, cpu_device, seed):
        """Test that dropout is applied in training mode."""
        torch.manual_seed(seed)
        embed_dim = 64
        layer = SelfAttentionLayer(embed_dim=embed_dim, dropout=0.5)
        layer = layer.to(cpu_device)
        layer.train()

        x = torch.ones(2, 32, embed_dim, device=cpu_device)
        output1 = layer(x)
        output2 = layer(x)

        assert not torch.allclose(output1, output2)

    def test_dropout_disabled_in_eval_mode(self, cpu_device, seed):
        """Test that dropout is disabled in evaluation mode."""
        torch.manual_seed(seed)
        embed_dim = 64
        layer = SelfAttentionLayer(embed_dim=embed_dim, dropout=0.5)
        layer = layer.to(cpu_device)
        layer.eval()

        x = torch.ones(2, 32, embed_dim, device=cpu_device)
        output1 = layer(x)
        output2 = layer(x)

        assert torch.allclose(output1, output2)

    def test_key_padding_mask(self, cpu_device):
        """Test that key padding mask works correctly."""
        embed_dim = 64
        layer = SelfAttentionLayer(embed_dim=embed_dim, dropout=0.0)
        layer = layer.to(cpu_device)

        batch_size = 2
        seq_len = 32
        x = torch.ones(batch_size, seq_len, embed_dim, device=cpu_device)

        # Create mask: first half of sequence is valid, second half is padding
        key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=cpu_device)
        key_padding_mask[:, seq_len // 2:] = True

        output = layer(x, key_padding_mask=key_padding_mask)
        assert output.shape == x.shape

    def test_attn_mask_causal(self, cpu_device):
        """Test causal (triangular) attention mask."""
        embed_dim = 64
        layer = SelfAttentionLayer(embed_dim=embed_dim, dropout=0.0)
        layer = layer.to(cpu_device)

        batch_size = 2
        seq_len = 32
        x = torch.ones(batch_size, seq_len, embed_dim, device=cpu_device)

        # Create causal mask: upper triangular is masked
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=cpu_device), diagonal=1)

        output = layer(x, attn_mask=attn_mask)
        assert output.shape == x.shape


# ============================================================================
# Tests for CrossAttentionLayer
# ============================================================================

class TestCrossAttentionLayer:
    """Tests for cross-attention layer."""

    def test_output_shape_matches_query(self, cpu_device):
        """Test that output shape matches query input shape."""
        embed_dim = 128
        num_heads = 4
        layer = CrossAttentionLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0)
        layer = layer.to(cpu_device)

        batch_size = 2
        query_len = 32
        kv_len = 64
        query = torch.randn(batch_size, query_len, embed_dim, device=cpu_device)
        key_value = torch.randn(batch_size, kv_len, embed_dim, device=cpu_device)

        output = layer(query, key_value)

        assert output.shape == query.shape

    def test_different_query_kv_lengths(self, cpu_device):
        """Test with different query and key/value lengths."""
        embed_dim = 64
        layer = CrossAttentionLayer(embed_dim=embed_dim, dropout=0.0)
        layer = layer.to(cpu_device)

        query = torch.randn(2, 20, embed_dim, device=cpu_device)
        key_value = torch.randn(2, 100, embed_dim, device=cpu_device)

        output = layer(query, key_value)

        assert output.shape[1] == 20  # Output length = query length

    def test_gradient_flow(self, cpu_device):
        """Test that gradients flow through the layer."""
        embed_dim = 64
        layer = CrossAttentionLayer(embed_dim=embed_dim, dropout=0.0)
        layer = layer.to(cpu_device)
        layer.train()

        query = torch.randn(2, 32, embed_dim, device=cpu_device, requires_grad=True)
        key_value = torch.randn(2, 32, embed_dim, device=cpu_device, requires_grad=True)
        output = layer(query, key_value)
        loss = output.sum()
        loss.backward()

        assert query.grad is not None
        assert query.grad.shape == query.shape

    def test_bidirectional_attention_pattern(self, cpu_device):
        """Test bidirectional attention pattern (target <-> off-target)."""
        embed_dim = 64
        layer = CrossAttentionLayer(embed_dim=embed_dim, dropout=0.0)
        layer = layer.to(cpu_device)

        # Simulate target attending to off-target
        target = torch.randn(2, 25, embed_dim, device=cpu_device)
        off_target_center = torch.randn(2, 23, embed_dim, device=cpu_device)

        # Target attends to off-target center
        target_updated = layer(target, off_target_center)
        assert target_updated.shape == target.shape

        # Off-target center attends to target (reverse direction)
        off_target_updated = layer(off_target_center, target)
        assert off_target_updated.shape == off_target_center.shape


# ============================================================================
# Tests for CRISCross (Main Model)
# ============================================================================

class TestCRISCross:
    """Tests for the main CRISCross transformer model."""

    def _get_minimal_config(self):
        """Get a minimal configuration for testing."""
        return {
            "vocab_size": 5,
            "dropout": 0.0,
            "context_layers": 2,
            "hidden_dim": 64,
            "num_epi": 0,
            "output_size": 1,
            "windowsize": 128,
            "merge": None,
        }

    def test_sequence_only_forward_pass(self, cpu_device):
        """Test forward pass with sequence-only model (no epigenetics)."""
        config = self._get_minimal_config()
        model = CRISCross(**config)
        model = model.to(cpu_device)
        model.eval()

        batch_size = 2
        target_seq_len = 25  # 24nt + CLS
        windowsize = 128

        target = torch.randint(0, 5, (batch_size, target_seq_len), device=cpu_device)
        off_target = torch.randint(0, 5, (batch_size, windowsize), device=cpu_device)
        strand = torch.randint(0, 2, (batch_size,), device=cpu_device)

        with torch.no_grad():
            cls_logits, transformer_outputs = model(target, off_target, strand, epi=None)

        assert cls_logits.shape == (batch_size, 1)
        # Note: transformer_outputs has same seq_len as input (CLS prepended after k-mer conversion)
        assert transformer_outputs.shape == (batch_size, target_seq_len, 64)

    def test_with_epigenetic_features(self, cpu_device):
        """Test forward pass with epigenetic features (early fusion)."""
        config = self._get_minimal_config()
        config["num_epi"] = 2
        config["merge"] = "early"
        model = CRISCross(**config)
        model = model.to(cpu_device)
        model.eval()

        batch_size = 2
        target_seq_len = 25
        windowsize = 128

        target = torch.randint(0, 5, (batch_size, target_seq_len), device=cpu_device)
        off_target = torch.randint(0, 5, (batch_size, windowsize), device=cpu_device)
        strand = torch.randint(0, 2, (batch_size,), device=cpu_device)
        # Epi features: [batch, num_epi, windowsize] - the Linear layer expects num_epi as last dim
        # So we need to transpose: [batch, windowsize, num_epi]
        epi = torch.randn(batch_size, windowsize, 2, device=cpu_device)

        with torch.no_grad():
            cls_logits, transformer_outputs = model(target, off_target, strand, epi=epi)

        assert cls_logits.shape == (batch_size, 1)
        assert transformer_outputs.shape == (batch_size, target_seq_len, 64)

    def test_gradient_computation(self, cpu_device):
        """Test that gradients can be computed through the model."""
        config = self._get_minimal_config()
        model = CRISCross(**config)
        model = model.to(cpu_device)
        model.train()

        batch_size = 2
        target_seq_len = 25
        windowsize = 128

        target = torch.randint(0, 5, (batch_size, target_seq_len), device=cpu_device, requires_grad=False)
        off_target = torch.randint(0, 5, (batch_size, windowsize), device=cpu_device, requires_grad=False)
        strand = torch.randint(0, 2, (batch_size,), device=cpu_device)

        cls_logits, transformer_outputs = model(target, off_target, strand, epi=None)
        loss = cls_logits.sum()
        loss.backward()

        # Check that main model parameters have gradients (some embeddings may not be used)
        grad_found = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_found = True
                break
        assert grad_found, "No parameters received gradients"

    def test_transformer_output_shape(self, cpu_device):
        """Test that transformer output shape matches documentation."""
        config = self._get_minimal_config()
        model = CRISCross(**config)
        model = model.to(cpu_device)
        model.eval()

        batch_size = 2
        target_seq_len = 25  # 24nt + CLS
        windowsize = 128

        target = torch.randint(0, 5, (batch_size, target_seq_len), device=cpu_device)
        off_target = torch.randint(0, 5, (batch_size, windowsize), device=cpu_device)
        strand = torch.randint(0, 2, (batch_size,), device=cpu_device)

        with torch.no_grad():
            cls_logits, transformer_outputs = model(target, off_target, strand, epi=None)

        # Transformer outputs should have same length as target input
        # (CLS is prepended after k-mer conversion, not removed from output)
        assert transformer_outputs.shape == (batch_size, target_seq_len, config["hidden_dim"])

    def test_kmer_conversion_pipeline(self, cpu_device):
        """Test that k-mer conversion works correctly in the full model."""
        config = self._get_minimal_config()
        model = CRISCross(**config)
        model = model.to(cpu_device)
        model.eval()

        # Use a known input pattern
        batch_size = 1
        target = torch.ones(batch_size, 25, device=cpu_device, dtype=torch.long)  # All 1s
        off_target = torch.ones(batch_size, 128, device=cpu_device, dtype=torch.long)  # All 1s
        strand = torch.zeros(batch_size, device=cpu_device, dtype=torch.long)

        with torch.no_grad():
            cls_logits, transformer_outputs = model(target, off_target, strand, epi=None)

        # Should produce valid outputs without errors
        assert not torch.isnan(cls_logits).any()
        assert not torch.isnan(transformer_outputs).any()

    def test_different_configurations(self, cpu_device):
        """Test model with different hyperparameter configurations."""
        configs = [
            {
                "vocab_size": 5,
                "dropout": 0.1,
                "context_layers": 1,
                "hidden_dim": 32,
                "num_epi": 0,
                "output_size": 1,
                "windowsize": 64,
                "merge": None,
            },
            {
                "vocab_size": 5,
                "dropout": 0.2,
                "context_layers": 3,
                "hidden_dim": 128,
                "num_epi": 2,
                "output_size": 1,
                "windowsize": 256,
                "merge": "early",
            },
        ]

        for config in configs:
            model = CRISCross(**config)
            model = model.to(cpu_device)
            model.eval()

            batch_size = 2
            target_seq_len = 25
            windowsize = config["windowsize"]

            target = torch.randint(0, 5, (batch_size, target_seq_len), device=cpu_device)
            off_target = torch.randint(0, 5, (batch_size, windowsize), device=cpu_device)
            strand = torch.randint(0, 2, (batch_size,), device=cpu_device)

            if config["merge"] == "early":
                # Epi features: [batch, windowsize, num_epi] for Linear layer
                epi = torch.randn(batch_size, windowsize, config["num_epi"], device=cpu_device)
            else:
                epi = None

            with torch.no_grad():
                cls_logits, transformer_outputs = model(target, off_target, strand, epi=epi)

            assert cls_logits.shape == (batch_size, config["output_size"])
            assert transformer_outputs.shape[2] == config["hidden_dim"]

    def test_batch_size_one(self, cpu_device):
        """Test with batch size of 1 (edge case)."""
        config = self._get_minimal_config()
        model = CRISCross(**config)
        model = model.to(cpu_device)
        model.eval()

        target = torch.randint(0, 5, (1, 25), device=cpu_device)
        off_target = torch.randint(0, 5, (1, 128), device=cpu_device)
        strand = torch.tensor([0], device=cpu_device)

        with torch.no_grad():
            cls_logits, transformer_outputs = model(target, off_target, strand, epi=None)

        assert cls_logits.shape == (1, 1)
        # Transformer outputs have same seq_len as input (CLS prepended after k-mer conversion)
        assert transformer_outputs.shape == (1, 25, 64)

    def test_output_range_sigmoid(self, cpu_device):
        """Test that sigmoid produces valid probabilities."""
        config = self._get_minimal_config()
        model = CRISCross(**config)
        model = model.to(cpu_device)
        model.eval()

        batch_size = 4
        target = torch.randint(0, 5, (batch_size, 25), device=cpu_device)
        off_target = torch.randint(0, 5, (batch_size, 128), device=cpu_device)
        strand = torch.randint(0, 2, (batch_size,), device=cpu_device)

        with torch.no_grad():
            cls_logits, _ = model(target, off_target, strand, epi=None)
            probs = torch.sigmoid(cls_logits)

        # Sigmoid output should be in [0, 1]
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_default_band_matches_the_legacy_slice(self, cpu_device):
        """band_delta=0 must reproduce the original centred 23-nt band exactly."""
        for windowsize in [64, 128, 256, 512, 511]:
            config = self._get_minimal_config()
            config["windowsize"] = windowsize
            model = CRISCross(**config)
            center = windowsize // 2 + windowsize % 2
            assert model.band_start == center - 23 // 2 - 1
            assert model.band_end == center + 23 // 2
            assert model.band_width == 23

    def test_widened_band_stays_pam_anchored(self, cpu_device):
        """Extra width is added PAM-distally; the PAM end never moves."""
        config = self._get_minimal_config()
        base = CRISCross(**config)
        for delta in [1, 2, 3]:
            config = self._get_minimal_config()
            config["band_delta"] = delta
            model = CRISCross(**config)
            assert model.band_end == base.band_end
            assert model.band_width == 23 + 2 * delta
            assert model.band_start == base.band_start - 2 * delta

    def test_forward_pass_with_widened_band(self, cpu_device):
        """A wider band must not change any tensor shape the caller sees."""
        for delta in [0, 1, 2, 3]:
            config = self._get_minimal_config()
            config["band_delta"] = delta
            model = CRISCross(**config).to(cpu_device)
            model.eval()

            target = torch.randint(0, 5, (2, 23), device=cpu_device)
            off_target = torch.randint(0, 5, (2, 128), device=cpu_device)
            strand = torch.tensor([0, 1], device=cpu_device)

            with torch.no_grad():
                cls_logits, outputs = model(target, off_target, strand, epi=None)

            assert cls_logits.shape == (2, 1)
            assert outputs.shape == (2, 23, config["hidden_dim"])

    def test_band_must_fit_in_the_window(self, cpu_device):
        config = self._get_minimal_config()
        config["windowsize"] = 24
        config["band_delta"] = 3
        with pytest.raises(ValueError, match="does not fit"):
            CRISCross(**config)

    def test_multiple_output_sizes(self, cpu_device):
        """Test with different output sizes."""
        for output_size in [1, 2, 3]:
            config = self._get_minimal_config()
            config["output_size"] = output_size
            model = CRISCross(**config)
            model = model.to(cpu_device)
            model.eval()

            target = torch.randint(0, 5, (2, 25), device=cpu_device)
            off_target = torch.randint(0, 5, (2, 128), device=cpu_device)
            strand = torch.tensor([0, 1], device=cpu_device)

            with torch.no_grad():
                cls_logits, _ = model(target, off_target, strand, epi=None)

            assert cls_logits.shape == (2, output_size)
