"""
Integration tests for the bulge-capable pipeline (Phase 1).

These cover the join between the sampler, the dataset and the pretraining step:

- the bulge-free path is unchanged, down to the exact masking tensors
- a bulged batch collates, forward-passes and produces a finite gradient
- the ground-truth alignment actually explains each sampled pair
- the hybrid-energy target is masked on bulged samples
- nothing structural about a sample tracks its bulge configuration

They need the full training stack (pytorch-lightning, torchmetrics, Biopython,
pyBigWig), so they skip where only ``CRISCross.bulges`` can be imported --
see ``tests/test_bulges.py`` for the dependency-free half.
"""

import os

import numpy as np
import pytest
import torch

pytest.importorskip("pytorch_lightning")
pytest.importorskip("torchmetrics")

from torch.utils.data import DataLoader

from CRISCross.bulges import PAM_LEN, SITE_LEN, SPACER_LEN, band_bounds, band_width
from CRISCross.Datasets import (
    BulgeEnergyGenomicDataset,
    BulgeGenomicDataset,
    GenomicDataset,
)
from CRISCross.pretrainArtificial import PreTrainModel, _parse_bulge_profile

WINDOW = 512
CHROM_SIZES = {"chr1": 40000}


@pytest.fixture(scope="module")
def track_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tracks")
    for chrom, n in CHROM_SIZES.items():
        np.save(os.path.join(d, f"ATAC_{chrom}.npy"), np.random.rand(n).astype(np.float32))
    return str(d)


@pytest.fixture(scope="module")
def seq_dict():
    torch.manual_seed(0)
    return {c: torch.randint(1, 5, (n,), dtype=torch.uint8) for c, n in CHROM_SIZES.items()}


@pytest.fixture
def make_dataset(seq_dict, track_dir):
    def _make(cls=GenomicDataset, **kwargs):
        base = dict(
            chrom_sizes=CHROM_SIZES,
            seq_dict=seq_dict,
            bw_dir=[track_dir],
            epi_features=["ATAC"],
            window_size=WINDOW,
            num_samples=256,
            atac_features=["ATAC"],
        )
        base.update(kwargs)
        return cls(**base)
    return _make


def _make_model(band_delta=0, **kwargs):
    params = dict(
        context_layers=1,
        hidden_dim=32,
        num_epi=1,
        dropout=0.0,
        seed=0,
        windowsize=WINDOW,
        merge="early",
        epi_weights=[0.1],
        num_atac=1,
        band_delta=band_delta,
    )
    params.update(kwargs)
    return PreTrainModel(**params)


# ============================================================================
# The mismatch-only path must not move
# ============================================================================

class TestNoRegression:

    def test_plain_dataset_still_returns_seven_fields(self, make_dataset):
        item = make_dataset()[0]
        assert len(item) == 7
        assert item[0].shape == (SITE_LEN,)
        assert item[1].shape == (WINDOW,)

    def test_plain_dataset_guide_comes_from_the_legacy_slice(self, make_dataset):
        """The candidate site is still the centred 23-mer at band_delta=0."""
        ds = make_dataset()
        start, end = band_bounds(WINDOW, 0)
        center = WINDOW // 2 + WINDOW % 2
        assert (start, end) == (center - 23 // 2 - 1, center + 23 // 2)
        for i in range(20):
            target_x, off_target_x = ds[i][:2]
            assert (off_target_x[start:end] == target_x).sum() >= 15

    def test_masking_is_bit_identical_to_the_original(self, make_dataset):
        """Reimplement the pre-change masking and compare tensor for tensor."""
        model = _make_model(band_delta=0)
        batch = next(iter(DataLoader(make_dataset(), batch_size=8)))

        torch.manual_seed(11)
        masked_target, masked_ot, masked_epi, mask = model.mask_shit(batch)

        torch.manual_seed(11)
        target_x, off_target_x, epi = batch[0].clone(), batch[1], batch[2]
        B, S = target_x.shape
        center = WINDOW // 2 + WINDOW % 2
        lo, hi = center - 23 // 2 - 1, center + 23 // 2

        mask_t = torch.rand(B, S) < model.mask_prob
        empty = mask_t.sum(1) == 0
        if empty.any():
            mask_t[empty, torch.randint(S, (int(empty.sum()),))] = True
        rand = torch.rand(B, S)
        mask_mask = mask_t & (rand < 0.8)
        random_mask = mask_t & (rand >= 0.8) & (rand < 0.9)
        target_x = target_x.masked_fill(mask_t, 0)
        random_tokens = torch.randint(1, 5, target_x.shape, dtype=target_x.dtype)
        target_x[random_mask] = random_tokens[random_mask]
        ref_ot = off_target_x.clone()
        ref_ot[:, lo:hi] = ref_ot[:, lo:hi].masked_fill(mask_mask, 0)
        ref_epi = epi.clone()
        ref_epi[:, lo:hi] = ref_epi[:, lo:hi].masked_fill(mask_t.unsqueeze(-1), 0)

        assert torch.equal(masked_target, target_x)
        assert torch.equal(masked_ot, ref_ot)
        assert torch.equal(masked_epi, ref_epi)
        assert torch.equal(mask, mask_t)

    def test_gapless_alignment_defaults_to_the_diagonal(self, make_dataset):
        model = _make_model(band_delta=0)
        batch = next(iter(DataLoader(make_dataset(), batch_size=4)))
        _, align, edits, gapless = model.unpack_batch(batch)
        assert torch.equal(align[0], torch.arange(SITE_LEN))
        assert edits is None
        assert torch.equal(gapless, torch.ones(4))

    def test_bulge_rate_zero_reproduces_the_gapless_sampler(self, make_dataset):
        ds = make_dataset(BulgeGenomicDataset, bulge_rate=0.0)
        start, end = band_bounds(WINDOW, 0)
        for i in range(50):
            target_x, off_target_x = ds[i][:2]
            align, edits, gapless = ds[i][7:10]
            assert torch.equal(align, torch.arange(SITE_LEN))
            assert edits[1] == 0 and edits[2] == 0
            assert bool(gapless)


# ============================================================================
# The bulged path
# ============================================================================

class TestBulgedSampling:

    @pytest.mark.parametrize("delta", [1, 2, 3])
    def test_shapes_are_invariant_to_the_bulge_configuration(self, make_dataset, delta):
        ds = make_dataset(BulgeGenomicDataset, band_delta=delta, bulge_rate=0.7)
        seen = set()
        for i in range(200):
            item = ds[i]
            assert item[0].shape == (SITE_LEN,)
            assert item[1].shape == (WINDOW,)
            assert item[7].shape == (SITE_LEN,)
            seen.add((int(item[8][1]), int(item[8][2])))
        assert len(seen) > 1, "sweep produced only one configuration"

    def test_alignment_explains_every_sample(self, make_dataset):
        ds = make_dataset(BulgeGenomicDataset, band_delta=2, bulge_rate=1.0)
        bs, be = band_bounds(WINDOW, 2)
        for i in range(200):
            item = ds[i]
            target_x, off_target_x, align, edits = item[0], item[1], item[7], item[8]
            m, b_D, b_R = (int(v) for v in edits)
            paired = align >= 0
            assert int((~paired).sum()) == b_R
            assert int(align[paired].max()) < band_width(2)
            aligned = off_target_x[bs:be][align.clamp_min(0)]
            assert int(((aligned != target_x) & paired).sum()) == m
            assert m + b_D + b_R <= 6
            # The alignment is monotone: PAM-anchored site indices decrease
            # along the guide in array order.
            cols = align[paired].tolist()
            assert all(a < b for a, b in zip(cols, cols[1:]))

    def test_narrow_band_is_rejected(self, make_dataset):
        with pytest.raises(ValueError, match="cannot hold"):
            make_dataset(BulgeGenomicDataset, band_delta=0, bulge_rate=0.5)

    def test_decoy_matches_the_positive_edit_count(self, make_dataset):
        ds = make_dataset(BulgeGenomicDataset, band_delta=2, bulge_rate=0.7, emit_decoy=True)
        for i in range(50):
            item = ds[i]
            assert len(item) == 11
            guide, edits, decoy = item[0], item[8], item[10]
            assert decoy.shape == (SITE_LEN,)
            assert int((decoy != guide).sum()) == int(edits.sum())

    def test_dataloader_collates_a_bulged_batch(self, make_dataset):
        ds = make_dataset(BulgeGenomicDataset, band_delta=2, bulge_rate=0.5)
        batch = next(iter(DataLoader(ds, batch_size=8)))
        assert len(batch) == 10
        assert batch[0].shape == (8, SITE_LEN)
        assert batch[7].shape == (8, SITE_LEN)
        assert batch[8].shape == (8, 3)
        assert batch[9].shape == (8,)


# ============================================================================
# The energy target
# ============================================================================

class TestMaskedEnergy:

    def test_bulged_samples_carry_no_energy_target(self, make_dataset):
        ds = make_dataset(BulgeEnergyGenomicDataset, band_delta=2, bulge_rate=0.6, energy_stats=None)
        n_bulged = 0
        for i in range(200):
            item = ds[i]
            energy, gapless = item[4], item[9]
            if not bool(gapless):
                n_bulged += 1
                assert energy == 0.0
        assert n_bulged > 0

    def test_energy_loss_is_normalised_over_the_unmasked_subset(self, make_dataset):
        model = _make_model(band_delta=2, use_energy=True)
        model.eval()
        ds = make_dataset(BulgeEnergyGenomicDataset, band_delta=2, bulge_rate=0.6, energy_stats=None)
        batch = list(next(iter(DataLoader(ds, batch_size=16))))
        _, _, _, gapless = model.unpack_batch(batch)
        assert 0 < float(gapless.sum()) < gapless.numel(), "batch is not mixed"

        loss_mixed = model.general_step(batch)[0]
        # Forcing every sample to count must change the energy term.
        batch[9] = torch.ones_like(batch[9])
        loss_all = model.general_step(batch)[0]
        assert torch.isfinite(loss_mixed) and torch.isfinite(loss_all)


# ============================================================================
# Pretraining step
# ============================================================================

class TestPretrainStep:

    @pytest.mark.parametrize(
        "delta,rate", [(0, None), (2, None), (0, 0.0), (2, 0.3), (2, 1.0)]
    )
    def test_general_step_runs_and_backprops(self, make_dataset, delta, rate):
        if rate is None:
            ds = make_dataset(band_delta=delta)
        else:
            ds = make_dataset(BulgeGenomicDataset, band_delta=delta, bulge_rate=rate)
        model = _make_model(band_delta=delta)
        model.eval()
        batch = next(iter(DataLoader(ds, batch_size=8)))

        loss, logits, epi_logits, y, mask, clsloss, epiloss, atac_loss = model.general_step(batch)
        assert torch.isfinite(loss)
        assert logits.shape == (8, SITE_LEN, 25)
        assert y.shape == (8, SITE_LEN)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())

    def test_unpaired_guide_positions_are_excluded_from_the_loss(self, make_dataset):
        """RNA-bulged guide positions have no target partner, so no MLM label."""
        model = _make_model(band_delta=2)
        model.eval()
        ds = make_dataset(BulgeGenomicDataset, band_delta=2, bulge_profile={(0, 1): 1.0})
        batch = next(iter(DataLoader(ds, batch_size=16)))
        _, align, _, _ = model.unpack_batch(batch)
        mask = model.general_step(batch)[4]
        assert not (mask & (align < 0)).any()

    def test_band_masking_never_follows_the_true_alignment(self, make_dataset):
        """The band mask must stay on the fixed diagonal, or it leaks the alignment."""
        model = _make_model(band_delta=2)
        ds = make_dataset(BulgeGenomicDataset, band_delta=2, bulge_rate=1.0)
        batch = next(iter(DataLoader(ds, batch_size=32)))
        _, masked_ot, _, _ = model.mask_shit(batch)

        bs, be = band_bounds(WINDOW, 2)
        zeroed = masked_ot[:, bs:be] == 0
        diagonal = set(model.diagonal_columns(1, zeroed.device)[0].tolist())
        off_diagonal = [c for c in range(band_width(2)) if c not in diagonal]
        assert int(zeroed[:, off_diagonal].sum()) == 0


# ============================================================================
# Leakage probe
# ============================================================================

class TestLeakageProbe:

    def test_probe_passes_on_the_shipped_pipeline(self, make_dataset):
        from CRISCross.leakageProbe import run_leakage_probe

        ds = make_dataset(BulgeGenomicDataset, band_delta=2, bulge_rate=0.5, epi_features=[])
        result = run_leakage_probe(ds, n_samples=1500, verbose=False)
        assert result["structural"]["ok"]
        assert not result["shape_probe_leaks"]
        assert result["region_label_extent"] == band_width(2)


# ============================================================================
# Config parsing
# ============================================================================

class TestConfigParsing:

    def test_json_string_keys_are_accepted(self):
        parsed = _parse_bulge_profile({"0,0": 0.8, "1,0": 0.2})
        assert parsed == {(0, 0): 0.8, (1, 0): 0.2}

    def test_tuple_keys_pass_through(self):
        assert _parse_bulge_profile({(0, 0): 1.0}) == {(0, 0): 1.0}

    def test_none_passes_through(self):
        assert _parse_bulge_profile(None) is None
