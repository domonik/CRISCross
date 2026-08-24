"""
Pytest tests for CRISCross/bulges.py.

The tests are grouped around the three things Phase 1 has to get exactly right:

1. **Geometry** -- the PAM-anchored band must reduce to the legacy centred
   23-nt slice when ``band_delta == 0``, otherwise the mismatch-only model
   silently changes behaviour.
2. **The length invariant and the alignment** -- the guide is always 20 + 3 nt,
   the protospacer is ``20 + b_D - b_R``, and the recorded alignment must
   actually reconstruct the pairing that was used to build the guide.
3. **Leakage** -- nothing about the sample's shape may depend on ``(b_D, b_R)``.

This module deliberately depends only on torch, so it runs without the heavy
pytorch-lightning / pyBigWig / Biopython stack that ``Datasets.py`` needs.
"""

import math

import pytest
import torch

from CRISCross.bulges import (
    DEFAULT_BULGE_PROFILE,
    MAX_BULGES,
    MAX_EDITS,
    PAM_LEN,
    SITE_LEN,
    SPACER_LEN,
    band_bounds,
    band_width,
    identity_alignment,
    legacy_band_bounds,
    make_bulged_guide,
    make_gapless_decoy,
    mismatch_count_weights,
    normalise_bulge_profile,
    pam_end_index,
    sample_bulge_config,
    site_index_to_band_index,
    structural_leakage_report,
)

ALL_CONFIGS = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]


def _legacy_slice(window_size):
    """The hard-coded band expression used throughout the original codebase."""
    center = window_size // 2 + window_size % 2
    return center - 23 // 2 - 1, center + 23 // 2


def _random_site(length, generator=None):
    return torch.randint(1, 5, (length,), generator=generator)


# ============================================================================
# Geometry
# ============================================================================

class TestGeometry:
    """The PAM-anchored band must be a strict generalisation of the old one."""

    @pytest.mark.parametrize("window_size", [64, 100, 101, 128, 256, 511, 512, 2000])
    def test_delta_zero_matches_legacy_slice(self, window_size):
        assert band_bounds(window_size, 0) == _legacy_slice(window_size)
        assert legacy_band_bounds(window_size) == _legacy_slice(window_size)

    @pytest.mark.parametrize("window_size", [128, 512])
    @pytest.mark.parametrize("delta", [0, 1, 2, 3])
    def test_band_is_pam_anchored(self, window_size, delta):
        """Widening the band must not move the PAM end -- only the 5' start."""
        _, legacy_end = _legacy_slice(window_size)
        start, end = band_bounds(window_size, delta)
        assert end == legacy_end
        assert end - start == band_width(delta) == SITE_LEN + 2 * delta

    def test_pam_end_index(self):
        assert pam_end_index(512) == 512 // 2 + 11

    def test_band_too_wide_raises(self):
        with pytest.raises(ValueError, match="does not fit"):
            band_bounds(20, 0)

    def test_negative_delta_raises(self):
        with pytest.raises(ValueError, match="band_delta"):
            band_width(-1)

    @pytest.mark.parametrize("delta", [0, 1, 2, 3])
    def test_identity_alignment_is_the_gapless_diagonal(self, delta):
        bw = band_width(delta)
        align = identity_alignment(bw)
        assert align.shape == (SITE_LEN,)
        # The PAM sits at the very end of the band for every delta.
        assert align[-1].item() == bw - 1
        assert align[SPACER_LEN - 1].item() == bw - 1 - PAM_LEN
        if delta == 0:
            assert torch.equal(align, torch.arange(SITE_LEN))

    def test_site_index_to_band_index_roundtrip(self):
        bw = band_width(2)
        align_site = torch.tensor([-1, 0, 5, 22])
        band = site_index_to_band_index(align_site, bw)
        assert band[0].item() == -1
        assert band[1].item() == bw - 1
        assert band[2].item() == bw - 6
        assert band[3].item() == bw - 23


# ============================================================================
# Bulge profile
# ============================================================================

class TestBulgeProfile:

    def test_default_profile_normalises(self):
        p = normalise_bulge_profile()
        assert pytest.approx(sum(p.values())) == 1.0
        assert p[(0, 0)] == pytest.approx(0.80)

    @pytest.mark.parametrize("rate", [0.0, 0.05, 0.2, 0.5, 1.0])
    def test_bulge_rate_sweep(self, rate):
        """The section 3.3 ablation: 0 / 5 / 20 / 50 / 100 % bulged."""
        p = normalise_bulge_profile(bulge_rate=rate)
        assert p[(0, 0)] == pytest.approx(1.0 - rate)
        assert sum(v for k, v in p.items() if k != (0, 0)) == pytest.approx(rate)
        # Relative proportions among the bulged entries are preserved.
        if rate > 0:
            assert p[(1, 0)] / p[(0, 1)] == pytest.approx(0.12 / 0.05)

    def test_bulge_rate_zero_is_mismatch_only(self):
        p = normalise_bulge_profile(bulge_rate=0.0)
        for _ in range(50):
            assert sample_bulge_config(p) == (0, 0)

    def test_bulge_rate_one_never_gapless(self):
        p = normalise_bulge_profile(bulge_rate=1.0)
        for _ in range(50):
            assert sample_bulge_config(p) != (0, 0)

    def test_rejects_over_budget_profile(self):
        with pytest.raises(ValueError, match="budget"):
            normalise_bulge_profile({(2, 1): 1.0})

    def test_rejects_negative_weights(self):
        with pytest.raises(ValueError, match="negative weight"):
            normalise_bulge_profile({(0, 0): 1.0, (1, 0): -0.5})

    def test_rejects_zero_total(self):
        with pytest.raises(ValueError, match="sum to zero"):
            normalise_bulge_profile({(0, 0): 0.0})

    def test_sample_covers_the_support(self):
        torch.manual_seed(0)
        seen = {sample_bulge_config(DEFAULT_BULGE_PROFILE) for _ in range(4000)}
        assert seen == set(ALL_CONFIGS)


# ============================================================================
# Mismatch count distribution
# ============================================================================

class TestMismatchCount:

    def test_matches_legacy_mutation_weights(self):
        """base_factor=3, 23 positions, n_max=6 is exactly the old MUT_WEIGHTS."""
        legacy = torch.tensor(
            [math.comb(23, i) * (3 ** i) for i in range(1, 7)], dtype=torch.double
        )
        legacy = legacy / legacy.sum()
        assert torch.allclose(mismatch_count_weights(23, 6, base_factor=3), legacy)

    def test_spec_form_is_position_only(self):
        w = mismatch_count_weights(20, 6, base_factor=1)
        expected = torch.tensor(
            [math.comb(20, i) for i in range(1, 7)], dtype=torch.double
        )
        assert torch.allclose(w, expected / expected.sum())

    def test_weights_sum_to_one(self):
        for n_pos in (17, 18, 19, 20, 21, 22, 23):
            for n_max in range(1, 7):
                w = mismatch_count_weights(n_pos, n_max)
                assert pytest.approx(float(w.sum())) == 1.0
                assert w.numel() == n_max

    def test_n_max_beyond_positions_raises(self):
        with pytest.raises(ValueError, match="exceeds"):
            mismatch_count_weights(3, 6)


# ============================================================================
# Guide construction, length invariant and alignment
# ============================================================================

class TestMakeBulgedGuide:

    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_length_invariant(self, config):
        """Guide is always 23; only the genomic site changes length."""
        torch.manual_seed(0)
        b_D, b_R = config
        L = SPACER_LEN + b_D - b_R
        site = _random_site(L + PAM_LEN)
        guide, align, m = make_bulged_guide(site, b_D, b_R)

        assert guide.shape == (SITE_LEN,)
        assert align.shape == (SITE_LEN,)
        assert 1 <= m <= MAX_EDITS - b_D - b_R
        assert m + b_D + b_R <= MAX_EDITS

    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_wrong_site_length_raises(self, config):
        b_D, b_R = config
        L = SPACER_LEN + b_D - b_R
        with pytest.raises(ValueError, match="site has"):
            make_bulged_guide(_random_site(L + PAM_LEN + 1), b_D, b_R)

    def test_illegal_config_raises(self):
        with pytest.raises(ValueError, match="illegal bulge configuration"):
            make_bulged_guide(_random_site(21), 2, 1)

    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_unpaired_count_equals_b_R(self, config):
        """Exactly b_R guide positions have no partner."""
        torch.manual_seed(1)
        b_D, b_R = config
        for _ in range(30):
            L = SPACER_LEN + b_D - b_R
            site = _random_site(L + PAM_LEN)
            _, align, _ = make_bulged_guide(site, b_D, b_R)
            assert int((align < 0).sum()) == b_R
            # The PAM is always paired, one-to-one, at the end of the site.
            assert align[SPACER_LEN:].tolist() == [2, 1, 0]

    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_alignment_is_a_valid_monotone_partial_matching(self, config):
        """No target base may be used twice, and the path may not cross itself."""
        torch.manual_seed(2)
        b_D, b_R = config
        for _ in range(30):
            L = SPACER_LEN + b_D - b_R
            site = _random_site(L + PAM_LEN)
            _, align, _ = make_bulged_guide(site, b_D, b_R)
            paired = align[align >= 0].tolist()
            assert len(paired) == len(set(paired)), "a target base was paired twice"
            # In array order the site index must be strictly decreasing
            # (both coordinate systems run in opposite directions).
            assert all(a > b for a, b in zip(paired, paired[1:])), "alignment crosses"
            # b_D target bases are consumed by nothing.
            spacer_used = {a for a in paired if a >= PAM_LEN}
            assert len(spacer_used) == SPACER_LEN - b_R
            assert L - len(spacer_used) == b_D

    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_alignment_explains_the_sequences(self, config):
        """Along the recorded alignment, mismatches must number exactly m."""
        torch.manual_seed(3)
        b_D, b_R = config
        for _ in range(50):
            L = SPACER_LEN + b_D - b_R
            site = _random_site(L + PAM_LEN)
            guide, align, m = make_bulged_guide(site, b_D, b_R)

            # PAM-anchored site index s -> array index of the site tensor.
            n_mismatch = 0
            for i in range(SITE_LEN):
                s = int(align[i])
                if s < 0:
                    continue
                site_arr = (L + PAM_LEN) - 1 - s
                if int(guide[i]) != int(site[site_arr]):
                    n_mismatch += 1
            assert n_mismatch == m

    def test_gapless_alignment_is_the_identity(self):
        """The no-regression case: a gapless sample aligns straight down."""
        torch.manual_seed(4)
        for _ in range(20):
            site = _random_site(SITE_LEN)
            _, align, _ = make_bulged_guide(site, 0, 0)
            # PAM-anchored site index of guide array index i is 22 - i.
            assert torch.equal(align, torch.arange(SITE_LEN).flip(0))
            assert torch.equal(
                site_index_to_band_index(align, SITE_LEN), identity_alignment(SITE_LEN)
            )

    def test_spacer_span_never_touches_the_pam(self):
        torch.manual_seed(5)
        for _ in range(50):
            site = _random_site(SITE_LEN)
            guide, _, _ = make_bulged_guide(site, 0, 0, mismatch_span="spacer")
            assert torch.equal(guide[SPACER_LEN:], site[SPACER_LEN:])

    def test_site_span_can_touch_the_pam(self):
        torch.manual_seed(6)
        touched = False
        for _ in range(200):
            site = _random_site(SITE_LEN)
            guide, _, _ = make_bulged_guide(site, 0, 0, mismatch_span="site")
            if not torch.equal(guide[SPACER_LEN:], site[SPACER_LEN:]):
                touched = True
                break
        assert touched, "mismatch_span='site' should occasionally mutate the PAM"

    def test_bad_span_raises(self):
        with pytest.raises(ValueError, match="mismatch_span"):
            make_bulged_guide(_random_site(SITE_LEN), 0, 0, mismatch_span="nope")

    def test_guide_bases_are_valid_tokens(self):
        torch.manual_seed(7)
        for config in ALL_CONFIGS:
            b_D, b_R = config
            L = SPACER_LEN + b_D - b_R
            guide, _, _ = make_bulged_guide(_random_site(L + PAM_LEN), b_D, b_R)
            assert int(guide.min()) >= 1 and int(guide.max()) <= 4

    def test_pam_distal_tilt_moves_bulges_away_from_the_pam(self):
        """The tilt option must actually shift the bulge position distribution."""
        torch.manual_seed(8)

        def mean_bulge_pam_index(tilt):
            total, n = 0.0, 0
            for _ in range(400):
                site = _random_site(SPACER_LEN + 1 + PAM_LEN)
                _, align, _ = make_bulged_guide(site, 1, 0, position_tilt=tilt)
                used = {int(a) for a in align if a >= PAM_LEN}
                skipped = [
                    s for s in range(PAM_LEN, PAM_LEN + SPACER_LEN + 1) if s not in used
                ]
                total += skipped[0]
                n += 1
            return total / n

        assert mean_bulge_pam_index(6.0) > mean_bulge_pam_index(0.0) + 1.0


# ============================================================================
# Homopolymer canonicalisation
# ============================================================================

class TestHomopolymerCanonicalisation:

    def test_dna_bulge_slides_to_the_pam_proximal_end_of_a_run(self):
        """A bulge in a poly-A run has no unique position; pick the leftmost."""
        torch.manual_seed(0)
        # Site of length 21 + 3: a run of A (token 1) in the middle, distinct
        # bases elsewhere so only the run is ambiguous.
        site = torch.tensor(
            [2, 3, 4, 2, 3, 4, 2, 3, 4, 1, 1, 1, 1, 1, 2, 3, 4, 2, 3, 4, 2] + [3, 4, 2],
            dtype=torch.long,
        )
        run_pam_indices = set()
        for i, tok in enumerate(site[:21]):
            if int(tok) == 1:
                run_pam_indices.add(PAM_LEN + (20 - i))  # 21-1-i, then +PAM_LEN

        seen_inside_run = 0
        for _ in range(400):
            _, align, _ = make_bulged_guide(
                site, 1, 0, mismatch_span="spacer", canonicalise_homopolymers=True
            )
            used = {int(a) for a in align if a >= PAM_LEN}
            skipped = [s for s in range(PAM_LEN, PAM_LEN + 21) if s not in used][0]
            if skipped in run_pam_indices:
                seen_inside_run += 1
                # Leftmost == PAM-proximal-most == smallest PAM-anchored index.
                assert skipped == min(run_pam_indices)
        assert seen_inside_run > 0, "the run was never hit -- test is not exercising anything"

    def test_canonicalisation_does_not_change_the_guide(self):
        """Canonicalisation is a relabelling: the sequences must be identical."""
        site = torch.tensor(
            [2, 3, 4, 2, 3, 4, 2, 3, 4, 1, 1, 1, 1, 1, 2, 3, 4, 2, 3, 4, 2] + [3, 4, 2],
            dtype=torch.long,
        )
        for seed in range(40):
            torch.manual_seed(seed)
            g_on, _, m_on = make_bulged_guide(site, 1, 0, canonicalise_homopolymers=True)
            torch.manual_seed(seed)
            g_off, _, m_off = make_bulged_guide(site, 1, 0, canonicalise_homopolymers=False)
            assert torch.equal(g_on, g_off)
            assert m_on == m_off

    def test_canonicalisation_keeps_the_alignment_consistent(self):
        """After sliding, the alignment must still explain the same mismatch count."""
        site = torch.tensor(
            [1, 1, 1, 1, 1, 1, 2, 3, 4, 1, 1, 1, 1, 1, 2, 3, 4, 2, 3, 4, 2] + [3, 4, 2],
            dtype=torch.long,
        )
        for seed in range(60):
            torch.manual_seed(seed)
            guide, align, m = make_bulged_guide(site, 1, 0)
            n = 0
            for i in range(SITE_LEN):
                s = int(align[i])
                if s < 0:
                    continue
                if int(guide[i]) != int(site[(21 + PAM_LEN) - 1 - s]):
                    n += 1
            assert n == m

    def test_rna_bulge_canonicalisation_preserves_the_guide(self):
        site = torch.tensor(
            [2, 3, 4, 2, 3, 4, 2, 3, 4, 1, 1, 1, 1, 2, 3, 4, 2, 3, 4] + [3, 4, 2],
            dtype=torch.long,
        )
        for seed in range(40):
            torch.manual_seed(seed)
            g_on, align_on, m_on = make_bulged_guide(site, 0, 1, canonicalise_homopolymers=True)
            torch.manual_seed(seed)
            g_off, _, m_off = make_bulged_guide(site, 0, 1, canonicalise_homopolymers=False)
            assert torch.equal(g_on, g_off)
            assert m_on == m_off
            assert int((align_on < 0).sum()) == 1


# ============================================================================
# Matched negatives
# ============================================================================

class TestGaplessDecoy:

    def test_decoy_has_the_requested_edit_count(self):
        torch.manual_seed(0)
        guide = _random_site(SITE_LEN)
        for k in range(0, 7):
            decoy = make_gapless_decoy(guide, k)
            assert decoy.shape == (SITE_LEN,)
            assert int((decoy != guide).sum()) == k

    def test_decoy_respects_the_spacer_span(self):
        torch.manual_seed(1)
        guide = _random_site(SITE_LEN)
        for _ in range(50):
            decoy = make_gapless_decoy(guide, 6, mismatch_span="spacer")
            assert torch.equal(decoy[SPACER_LEN:], guide[SPACER_LEN:])

    def test_decoy_matches_a_bulged_positive_edit_count(self):
        """A decoy for a (m, b_D, b_R) positive carries m + b_D + b_R mismatches."""
        torch.manual_seed(2)
        site = _random_site(SPACER_LEN + 1 + PAM_LEN)
        guide, _, m = make_bulged_guide(site, 1, 0)
        decoy = make_gapless_decoy(guide, m + 1)
        assert int((decoy != guide).sum()) == m + 1

    def test_bad_guide_length_raises(self):
        with pytest.raises(ValueError, match="guide must be"):
            make_gapless_decoy(_random_site(20), 3)


# ============================================================================
# Leakage (spec section 6.2)
# ============================================================================

class TestStructuralLeakage:

    def test_no_shape_depends_on_the_bulge_configuration(self):
        torch.manual_seed(0)
        window_size, delta = 512, 2
        bw = band_width(delta)
        start, end = band_bounds(window_size, delta)
        profile = normalise_bulge_profile(bulge_rate=0.6)

        samples = []
        for _ in range(300):
            window = torch.randint(1, 5, (window_size,))
            b_D, b_R = sample_bulge_config(profile)
            L = SPACER_LEN + b_D - b_R
            site = window[end - (L + PAM_LEN):end]
            guide, _, m = make_bulged_guide(site, b_D, b_R)
            samples.append(
                {
                    "guide": guide,
                    "window": window,
                    "band": window[start:end],
                    "edits": torch.tensor([m, b_D, b_R]),
                }
            )

        report = structural_leakage_report(samples)
        assert report["ok"], report
        assert report["guide_lengths"] == [SITE_LEN]
        assert report["window_lengths"] == [window_size]
        assert report["band_widths"] == [bw]
        assert len(report["bulge_configs"]) > 1, "sweep did not produce bulged samples"

    def test_report_flags_a_ragged_batch(self):
        samples = [
            {
                "guide": torch.zeros(SITE_LEN),
                "window": torch.zeros(512),
                "band": torch.zeros(23),
                "edits": torch.tensor([1, 0, 0]),
            },
            {
                "guide": torch.zeros(SITE_LEN),
                "window": torch.zeros(512),
                "band": torch.zeros(24),  # a band whose width tracks b_D: a leak
                "edits": torch.tensor([1, 1, 0]),
            },
        ]
        assert not structural_leakage_report(samples)["ok"]
