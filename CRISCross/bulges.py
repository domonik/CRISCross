"""
Bulge (indel) support for CRISCross / CRISPRAT.

This module implements the *data-side* half of the bulge extension: the geometry
of the PAM-anchored cross-attention band, and the CRISPRAT sampling procedure
that produces guide/target pairs containing DNA and RNA bulges together with the
ground-truth alignment between them.

Design commitment
-----------------
**Bulges are never encoded.** There are no gap tokens, no bulge-type embedding
and no unpaired-position label anywhere in the model input. A sample always
consists of a 23-nt guide (20-nt spacer + 3-nt PAM) and a contiguous genomic
window of fixed width. Everything a bulge does is a property of the *alignment*
between those two, which is the job of cross-attention to infer. The alignment
produced here is therefore a **training label only** -- it must never be fed to
the model as an input.

Terminology (Cas-OFFinder convention)
-------------------------------------
======  ==============  =====================================================
Symbol  Name            Meaning
======  ==============  =====================================================
``m``   mismatch        both sides paired but non-complementary
``b_D`` DNA bulge       unpaired base on the DNA side; protospacer is *longer*
``b_R`` RNA bulge       unpaired base on the guide side; protospacer is *shorter*
======  ==============  =====================================================

Length invariant::

    guide spacer length   = 20                     (always)
    protospacer length L  = 20 + b_D - b_R
    full target site      = L + 3 (PAM)

Budget: ``m + b_D + b_R <= 6`` and ``b_D + b_R <= 2``.

Coordinate conventions
----------------------
Two coordinate systems are used and it is worth being precise about both.

*Array order* is how sequences are stored everywhere else in this codebase:
5'->3' with the PAM in the last three positions::

    array index:  0  1  2 ... 19 | 20 21 22
                  <--- spacer --->  <- PAM ->

*PAM-anchored order* is how positions are sampled and reported, following the
Cas9 biology (nucleation at the PAM, unzipping PAM-distal) and the existing
``POS_WGH`` indexing in :mod:`CRISCross.energyCalculations`: index 0 is the base
immediately adjacent to the PAM and the index grows towards the PAM-distal 5'
end. For a 23-nt site the two are related by ``pam_index = 22 - array_index``,
so ``pam_index`` 0, 1, 2 are the three PAM bases themselves and ``pam_index`` 3
is the PAM-proximal-most spacer base.

The alignment returned by :func:`make_bulged_guide` is indexed by *guide array
index* and its values are *PAM-anchored site indices*, which makes the
conversion to a band column a single subtraction independent of the bulge
configuration (see :func:`site_index_to_band_index`).

No-regression guarantee
-----------------------
With ``band_delta=0`` and a gapless sample the band bounds returned by
:func:`band_bounds` are byte-identical to the hard-coded
``center - 23//2 - 1 : center + 23//2`` slice used throughout the original code,
and the alignment is exactly ``arange(23)``. Every downstream change is written
so that it collapses onto the original computation in that case.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPACER_LEN = 20
PAM_LEN = 3
SITE_LEN = SPACER_LEN + PAM_LEN  # 23

#: Maximum number of edits of any kind on a single sample (m + b_D + b_R).
MAX_EDITS = 6
#: Maximum number of bulges of any kind on a single sample (b_D + b_R).
MAX_BULGES = 2

#: A bulge configuration is the pair ``(b_D, b_R)``.
BulgeConfig = Tuple[int, int]

#: Default categorical table over bulge configurations. DNA bulges are more
#: common than RNA bulges in real data, and singles dominate doubles.
DEFAULT_BULGE_PROFILE: Dict[BulgeConfig, float] = {
    (0, 0): 0.80,  # gapless
    (1, 0): 0.12,  # one DNA bulge
    (0, 1): 0.05,  # one RNA bulge
    (2, 0): 0.01,
    (1, 1): 0.01,
    (0, 2): 0.01,
}

#: Token ids used for DNA bases (N=0 is never emitted by the sampler).
BASE_TOKENS = (1, 2, 3, 4)

MISMATCH_SPANS = ("site", "spacer")


# ---------------------------------------------------------------------------
# Window / band geometry
# ---------------------------------------------------------------------------

def pam_end_index(window_size: int) -> int:
    """Return the index one past the last PAM base of the candidate site.

    This is the single anchor from which every other coordinate is derived. It
    reproduces exactly the end of the legacy centred slice
    ``off_target_x[center - 23//2 - 1 : center + 23//2]`` where
    ``center = window_size // 2 + window_size % 2``.

    Args:
        window_size: Length of the genomic window handed to the model.

    Returns:
        Index one past the final PAM base, i.e. the exclusive end of the band.
    """
    center = window_size // 2 + window_size % 2
    return center + SITE_LEN // 2


def band_width(band_delta: int = 0) -> int:
    """Width of the cross-attention band.

    The band is always ``23 + 2 * band_delta`` nucleotides wide **regardless of
    the bulge configuration of the sample**. A window sized ``L + 3`` would
    announce ``(b_D, b_R)`` by its length alone, which is precisely the leak
    that must not exist.

    Args:
        band_delta: Half-width of the extra PAM-distal margin. ``0`` reproduces
            the original 23-nt band. ``2`` (27 nt) leaves room for protospacers
            of up to 22 nt (``b_D = 2``) plus slack for the offset bias.

    Returns:
        The band width in nucleotides.
    """
    if band_delta < 0:
        raise ValueError(f"band_delta must be >= 0, got {band_delta}")
    return SITE_LEN + 2 * band_delta


def band_bounds(window_size: int, band_delta: int = 0) -> Tuple[int, int]:
    """Return the ``(start, end)`` bounds of the PAM-anchored band.

    The band is anchored on the PAM and all extra width is added on the
    PAM-distal side, because the PAM is what is fixed in the genome: a DNA bulge
    lengthens the protospacer *away* from the PAM, never through it. Anchoring
    on the PAM keeps the alignment diagonal intact up to the bulge and merely
    shifted after it; a centred window would displace everything and be far
    harder to learn.

    Args:
        window_size: Length of the genomic window handed to the model.
        band_delta: See :func:`band_width`.

    Returns:
        ``(start, end)`` such that ``window[start:end]`` is the band.

    Raises:
        ValueError: If the band would run off the 5' edge of the window.
    """
    end = pam_end_index(window_size)
    start = end - band_width(band_delta)
    if start < 0:
        raise ValueError(
            f"band of width {band_width(band_delta)} does not fit in a window of "
            f"size {window_size} (start would be {start})"
        )
    return start, end


def legacy_band_bounds(window_size: int) -> Tuple[int, int]:
    """The original centred 23-nt band. Identical to ``band_bounds(w, 0)``."""
    return band_bounds(window_size, 0)


def site_index_to_band_index(align_site: torch.Tensor, bw: int) -> torch.Tensor:
    """Convert PAM-anchored site indices to band column indices.

    Both coordinate systems count from the PAM, in opposite directions, so the
    conversion is a single reflection. Entries equal to ``-1`` (unpaired guide
    positions, i.e. RNA bulges) are passed through unchanged.

    Args:
        align_site: Long tensor of PAM-anchored site indices, ``-1`` for
            unpaired.
        bw: Band width, from :func:`band_width`.

    Returns:
        Long tensor of the same shape holding band-relative column indices.
    """
    out = bw - 1 - align_site
    return torch.where(align_site < 0, torch.full_like(align_site, -1), out)


def identity_alignment(bw: int, device=None) -> torch.Tensor:
    """The gapless alignment: guide position *i* pairs band column *i + (bw-23)*.

    This is the correct alignment for every gapless sample at any band width,
    and reduces to ``arange(23)`` when ``band_delta == 0``. It is used as the
    fallback whenever a batch carries no ground-truth alignment, which makes the
    alignment-aware code paths collapse onto the original ones exactly.

    Args:
        bw: Band width.
        device: Optional torch device.

    Returns:
        Long tensor ``[23]`` of band column indices.
    """
    return torch.arange(SITE_LEN, device=device, dtype=torch.long) + (bw - SITE_LEN)


# ---------------------------------------------------------------------------
# Step 1 -- the bulge configuration
# ---------------------------------------------------------------------------

def normalise_bulge_profile(
    profile: Optional[Dict[BulgeConfig, float]] = None,
    bulge_rate: Optional[float] = None,
) -> Dict[BulgeConfig, float]:
    """Normalise a bulge profile, optionally forcing the gapless/bulged ratio.

    The bulge rate is deliberately a *hyperparameter* rather than something
    derived from a combinatorial formula, so that the gapless/bulged sweep
    (0% / 5% / 20% / 50% / 100%) can be run as a clean ablation.

    Args:
        profile: Mapping ``(b_D, b_R) -> weight``. Defaults to
            :data:`DEFAULT_BULGE_PROFILE`. Weights need not sum to 1.
        bulge_rate: If given, the returned profile has
            ``P(b_D = b_R = 0) = 1 - bulge_rate`` and the bulged entries
            rescaled to sum to ``bulge_rate``, preserving their relative
            proportions. ``0.0`` gives a purely mismatch-only sampler; ``1.0``
            gives the bulge-specialist upper bound.

    Returns:
        A new dict whose values sum to 1.

    Raises:
        ValueError: On invalid configurations, negative weights, or a request
            for ``bulge_rate > 0`` from a profile with no bulged entries.
    """
    profile = dict(DEFAULT_BULGE_PROFILE if profile is None else profile)
    profile = {(int(k[0]), int(k[1])): float(v) for k, v in profile.items()}

    for (b_D, b_R), w in profile.items():
        if b_D < 0 or b_R < 0:
            raise ValueError(f"negative bulge count in profile key {(b_D, b_R)}")
        if b_D + b_R > MAX_BULGES:
            raise ValueError(
                f"profile key {(b_D, b_R)} exceeds the budget b_D + b_R <= {MAX_BULGES}"
            )
        if b_R > SPACER_LEN:
            raise ValueError(f"profile key {(b_D, b_R)} would give a non-positive protospacer")
        if w < 0:
            raise ValueError(f"negative weight {w} for profile key {(b_D, b_R)}")

    if bulge_rate is not None:
        if not 0.0 <= bulge_rate <= 1.0:
            raise ValueError(f"bulge_rate must lie in [0, 1], got {bulge_rate}")
        bulged = {k: v for k, v in profile.items() if k != (0, 0)}
        bulged_total = sum(bulged.values())
        if bulge_rate > 0 and bulged_total <= 0:
            raise ValueError("bulge_rate > 0 requested but the profile has no bulged entries")
        profile = {(0, 0): 1.0 - bulge_rate}
        for k, v in bulged.items():
            profile[k] = bulge_rate * v / bulged_total if bulged_total > 0 else 0.0

    total = sum(profile.values())
    if total <= 0:
        raise ValueError("bulge profile weights sum to zero")
    return {k: v / total for k, v in profile.items()}


def sample_bulge_config(
    profile: Optional[Dict[BulgeConfig, float]] = None,
    generator: Optional[torch.Generator] = None,
) -> BulgeConfig:
    """Draw ``(b_D, b_R)`` from a (already normalised or not) bulge profile.

    Args:
        profile: Mapping ``(b_D, b_R) -> weight``; defaults to
            :data:`DEFAULT_BULGE_PROFILE`.
        generator: Optional torch generator for reproducible sampling.

    Returns:
        The sampled ``(b_D, b_R)``.
    """
    profile = normalise_bulge_profile(profile)
    keys = sorted(profile)
    weights = torch.tensor([profile[k] for k in keys], dtype=torch.double)
    idx = int(torch.multinomial(weights, 1, generator=generator).item())
    return keys[idx]


# ---------------------------------------------------------------------------
# Step 3 -- how many mismatches
# ---------------------------------------------------------------------------

def mismatch_count_weights(
    n_positions: int,
    n_max: int,
    base_factor: int = 3,
) -> torch.Tensor:
    """Normalised weights for ``P(N = n)``, ``n = 1..n_max``.

    ``weight(n) = C(n_positions, n) * base_factor ** n``.

    ``base_factor=3`` counts the three possible substitutions per position and
    reproduces the original ``mutation_weights`` in :mod:`CRISCross.Datasets`
    exactly (``n_positions=23``, ``n_max=6``). ``base_factor=1`` gives the
    position-only form written in the spec. The choice materially changes the
    mismatch-count distribution -- with ``base_factor=3`` almost all mass sits
    on 5-6 mismatches -- so it is exposed as a parameter rather than fixed.

    Args:
        n_positions: Number of positions a mismatch may be placed on.
        n_max: Largest permitted mismatch count.
        base_factor: Per-position multiplicity; 3 (legacy) or 1 (spec form).

    Returns:
        Double tensor of length ``n_max`` summing to 1, indexed by ``n - 1``.
    """
    if n_max < 1:
        raise ValueError(f"n_max must be >= 1, got {n_max}")
    if n_max > n_positions:
        raise ValueError(f"n_max={n_max} exceeds n_positions={n_positions}")
    weights = torch.tensor(
        [math.comb(n_positions, n) * (base_factor ** n) for n in range(1, n_max + 1)],
        dtype=torch.double,
    )
    return weights / weights.sum()


def sample_mismatch_count(
    n_positions: int,
    n_max: int,
    base_factor: int = 3,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Draw a mismatch count in ``1..n_max``. See :func:`mismatch_count_weights`."""
    weights = mismatch_count_weights(n_positions, n_max, base_factor)
    return int(torch.multinomial(weights, 1, generator=generator).item()) + 1


# ---------------------------------------------------------------------------
# Step 4 -- where the edits go
# ---------------------------------------------------------------------------

def _tilted_weights(n: int, tilt: float) -> torch.Tensor:
    """Linear PAM-distal tilt over ``n`` PAM-anchored positions.

    ``w(c) = 1 + tilt * c / (n - 1)`` with ``c = 0`` PAM-proximal. ``tilt = 0``
    is uniform; ``tilt = 3`` makes the PAM-distal end four times as likely as
    the PAM-proximal end. Real bulges are strongly PAM-distal, but for
    pretraining a broader distribution than reality is often preferable -- the
    prior gets corrected during fine-tuning -- which is why the default is
    uniform and the tilt is an ablation knob.
    """
    if n <= 0:
        return torch.zeros(0, dtype=torch.double)
    if n == 1 or tilt == 0.0:
        return torch.ones(n, dtype=torch.double)
    c = torch.arange(n, dtype=torch.double)
    w = 1.0 + tilt * c / (n - 1)
    return w.clamp_min(1e-12)


def _choose_without_replacement(
    k: int,
    weights: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> List[int]:
    """Draw ``k`` distinct indices proportional to ``weights``."""
    if k <= 0:
        return []
    if k > weights.numel():
        raise ValueError(f"cannot draw {k} distinct positions from {weights.numel()}")
    idx = torch.multinomial(weights, k, replacement=False, generator=generator)
    return [int(v) for v in idx]


# ---------------------------------------------------------------------------
# Steps 5-7 -- build the guide, record the alignment, canonicalise
# ---------------------------------------------------------------------------

def _walk_layout(layout: Sequence[str]) -> Tuple[List[Optional[int]], List[Optional[int]]]:
    """Walk an alignment-column layout from the PAM outwards.

    A layout is a list of column types in PAM-anchored order:

    - ``"P"`` -- paired: consumes one guide base and one target base
    - ``"D"`` -- DNA bulge: consumes one target base, no guide partner
    - ``"R"`` -- RNA bulge: consumes one guide base, no target partner

    The number of columns is ``20 + b_D``; ``#P = 20 - b_R``, so the walk
    consumes exactly ``20`` guide bases and ``L = 20 + b_D - b_R`` target bases.

    Args:
        layout: Column types, PAM-anchored order.

    Returns:
        ``(align, col_target)`` where ``align[g]`` is the PAM-anchored
        *protospacer* index paired with PAM-anchored guide index ``g`` (or
        ``None``), and ``col_target[c]`` is the protospacer index consumed by
        column ``c`` (or ``None`` for an ``"R"`` column).
    """
    align: List[Optional[int]] = []
    col_target: List[Optional[int]] = []
    ti = 0
    for typ in layout:
        if typ == "D":
            col_target.append(ti)
            ti += 1
        elif typ == "R":
            col_target.append(None)
            align.append(None)
        else:
            col_target.append(ti)
            align.append(ti)
            ti += 1
    return align, col_target


def _canonicalise_layout(
    layout: List[str],
    t_pam: Sequence[int],
    g_pam: Sequence[int],
) -> List[str]:
    """Slide bulges as PAM-proximal as possible without changing the sequences.

    A bulge inside a homopolymer run has no unique position: several alignments
    explain the same pair of sequences equally well. Left unhandled that injects
    label noise exactly where bulges are most common, so a canonical placement
    is chosen -- **leftmost, i.e. PAM-proximal-most**.

    A bulge column is swapped one step towards the PAM only when the swap leaves
    both sequences unchanged:

    - ``["P", "D"] -> ["D", "P"]`` moves the guide base of the ``P`` column from
      target base ``u`` to ``u + 1``; valid iff ``t_pam[u] == t_pam[u+1]``.
    - ``["P", "R"] -> ["R", "P"]`` swaps two adjacent guide bases; valid iff
      they are equal.

    Because the swap condition is exactly "the two bases involved are equal",
    both sequences are provably untouched and only the alignment label changes.

    Args:
        layout: Column types in PAM-anchored order (mutated copy returned).
        t_pam: Protospacer bases in PAM-anchored order.
        g_pam: Final guide spacer bases (post-mismatch) in PAM-anchored order.

    Returns:
        The canonicalised layout.
    """
    layout = list(layout)
    changed = True
    while changed:
        changed = False
        for c in range(1, len(layout)):
            if layout[c] == "P" or layout[c - 1] != "P":
                continue
            _, col_target = _walk_layout(layout)
            if layout[c] == "D":
                u = col_target[c - 1]  # target base paired at column c-1
                # column c consumes u + 1 by construction of the walk
                if u is not None and u + 1 < len(t_pam) and t_pam[u] == t_pam[u + 1]:
                    layout[c - 1], layout[c] = layout[c], layout[c - 1]
                    changed = True
            else:  # "R"
                # guide index paired at column c-1, and the unpaired one at c
                g = sum(1 for typ in layout[:c - 1] if typ in ("P", "R"))
                if g + 1 < len(g_pam) and g_pam[g] == g_pam[g + 1]:
                    layout[c - 1], layout[c] = layout[c], layout[c - 1]
                    changed = True
    return layout


def make_bulged_guide(
    site: torch.Tensor,
    b_D: int,
    b_R: int,
    *,
    mismatch_span: str = "site",
    mismatch_base_factor: int = 3,
    position_tilt: float = 0.0,
    mismatch_tilt: float = 0.0,
    canonicalise_homopolymers: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Build a guide from a genomic site, with ``b_D`` DNA and ``b_R`` RNA bulges.

    This is steps 3-7 of the CRISPRAT sampling procedure. The guide is always
    23 tokens (20-nt spacer + the site's own 3-nt PAM), whatever the bulge
    configuration; the *site* is what changes length.

    Both sequences are represented on the protospacer strand in the token
    alphabet of :mod:`CRISCross.Datasets` (A=1, C=2, G=3, T=4), so "paired"
    means *equal tokens* and a mismatch means a different token -- the same
    convention the rest of the codebase already uses.

    Args:
        site: Long tensor ``[L + 3]`` with ``L = 20 + b_D - b_R``, 5'->3', PAM
            last. Must not contain the N token (0).
        b_D: Number of DNA bulges (unpaired target bases).
        b_R: Number of RNA bulges (unpaired guide bases).
        mismatch_span: ``"site"`` lets mismatches fall on the 3 PAM positions as
            well as the spacer (the behaviour of the original ``mutate_target``,
            and therefore the default); ``"spacer"`` restricts them to the 20
            spacer positions as written in the spec.
        mismatch_base_factor: See :func:`mismatch_count_weights`.
        position_tilt: PAM-distal tilt for bulge positions; 0 is uniform.
        mismatch_tilt: PAM-distal tilt for mismatch positions; 0 is uniform.
        canonicalise_homopolymers: Apply the leftmost-placement rule of
            :func:`_canonicalise_layout`.
        generator: Optional torch generator.

    Returns:
        ``(guide, align_site, n_mismatch)``:

        - ``guide``: long tensor ``[23]`` in array order (5'->3', PAM last).
        - ``align_site``: long tensor ``[23]`` indexed by guide array index,
          holding the **PAM-anchored site index** of the paired target base, or
          ``-1`` where the guide base has no partner (an RNA bulge). Convert to
          band columns with :func:`site_index_to_band_index`. This is a label,
          never a model input.
        - ``n_mismatch``: the realised ``m``.

    Raises:
        ValueError: If the site length contradicts the length invariant, if the
            edit budget is exceeded, or on an unknown ``mismatch_span``.
    """
    if mismatch_span not in MISMATCH_SPANS:
        raise ValueError(f"mismatch_span must be one of {MISMATCH_SPANS}, got {mismatch_span!r}")
    if b_D < 0 or b_R < 0 or b_D + b_R > MAX_BULGES:
        raise ValueError(f"illegal bulge configuration ({b_D}, {b_R})")

    L = SPACER_LEN + b_D - b_R
    if site.numel() != L + PAM_LEN:
        raise ValueError(
            f"site has {site.numel()} tokens but ({b_D}, {b_R}) requires {L + PAM_LEN} "
            f"(L = 20 + b_D - b_R = {L}, plus {PAM_LEN} PAM)"
        )

    site = site.long()
    device = site.device
    # PAM-anchored views. t_pam[0] is the PAM-proximal-most protospacer base.
    t_pam: List[int] = [int(v) for v in site[:L].flip(0)]
    pam_tokens: List[int] = [int(v) for v in site[L:]]

    # --- Step 4a: place the bulges over the 20 + b_D alignment columns --------
    n_cols = SPACER_LEN + b_D
    n_bulges = b_D + b_R
    layout = ["P"] * n_cols
    if n_bulges:
        cols = _choose_without_replacement(
            n_bulges, _tilted_weights(n_cols, position_tilt), generator
        )
        for j, c in enumerate(cols):
            layout[c] = "D" if j < b_D else "R"

    align_pam, _ = _walk_layout(layout)

    # --- Step 5: read the guide out of the target ----------------------------
    g_pam: List[int] = []
    for g, t in enumerate(align_pam):
        if t is None:
            # RNA bulge: an extra guide base with no target partner.
            g_pam.append(_random_base(generator))
        else:
            g_pam.append(t_pam[t])

    # --- Step 3 + 4b: mismatches over the paired positions -------------------
    # Candidate guide array indices: every paired spacer position, plus the PAM
    # when mismatch_span == "site".
    paired_arr = [SPACER_LEN - 1 - g for g, t in enumerate(align_pam) if t is not None]
    cand = sorted(paired_arr)
    if mismatch_span == "site":
        cand += [SPACER_LEN, SPACER_LEN + 1, SPACER_LEN + 2]
    n_positions = len(cand)  # == 23 - b_R for "site", 20 - b_R for "spacer"
    n_max = min(MAX_EDITS - b_D - b_R, n_positions)
    n_mismatch = sample_mismatch_count(n_positions, n_max, mismatch_base_factor, generator)

    # Weight candidates by their PAM-anchored index over the whole 23-nt site.
    cand_pam = torch.tensor([SITE_LEN - 1 - i for i in cand], dtype=torch.double)
    if mismatch_tilt == 0.0:
        cand_w = torch.ones(n_positions, dtype=torch.double)
    else:
        cand_w = (1.0 + mismatch_tilt * cand_pam / (SITE_LEN - 1)).clamp_min(1e-12)
    chosen = _choose_without_replacement(n_mismatch, cand_w, generator)

    guide_arr = [0] * SITE_LEN
    for g, base in enumerate(g_pam):
        guide_arr[SPACER_LEN - 1 - g] = base
    guide_arr[SPACER_LEN:] = pam_tokens

    for ci in chosen:
        i = cand[ci]
        guide_arr[i] = _random_base(generator, exclude=guide_arr[i])

    # --- Step 7: canonicalise homopolymer bulges -----------------------------
    if canonicalise_homopolymers and n_bulges:
        g_pam_final = [guide_arr[SPACER_LEN - 1 - g] for g in range(SPACER_LEN)]
        layout = _canonicalise_layout(layout, t_pam, g_pam_final)
        align_pam, _ = _walk_layout(layout)

    # --- Step 6: emit the alignment ------------------------------------------
    align_site = [-1] * SITE_LEN
    for g, t in enumerate(align_pam):
        if t is not None:
            # PAM-anchored site index: PAM occupies 0..2, spacer starts at 3.
            align_site[SPACER_LEN - 1 - g] = PAM_LEN + t
    for k in range(PAM_LEN):
        # Guide PAM base at array index 20 + k has PAM-anchored index 2 - k.
        align_site[SPACER_LEN + k] = PAM_LEN - 1 - k

    return (
        torch.tensor(guide_arr, dtype=torch.long, device=device),
        torch.tensor(align_site, dtype=torch.long, device=device),
        n_mismatch,
    )


def _random_base(
    generator: Optional[torch.Generator] = None,
    exclude: Optional[int] = None,
) -> int:
    """Draw a random base token in 1..4, optionally excluding one."""
    while True:
        b = int(torch.randint(1, 5, (1,), generator=generator).item())
        if exclude is None or b != exclude:
            return b


# ---------------------------------------------------------------------------
# Matched negatives
# ---------------------------------------------------------------------------

def make_gapless_decoy(
    guide: torch.Tensor,
    n_mismatch: int,
    *,
    mismatch_span: str = "site",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Build a gapless 23-nt decoy site for a guide, with a matched edit count.

    The point of a matched negative is that it is *equally well explained by a
    gapless high-mismatch alignment* as the true site is by a gapped one.
    Deciding between those two explanations is the core skill the model has to
    acquire, so the decoy must not be distinguishable by edit count.

    Note that the decoy is synthetic rather than genomic: it is derived from the
    guide, not found in the genome. It therefore has **no genomic coordinate**
    and must not be paired with epigenetic tracks. It is intended for the
    sequence-only contrastive objective.

    Args:
        guide: Long tensor ``[23]``, the guide in array order.
        n_mismatch: Number of positions at which the decoy differs.
        mismatch_span: ``"site"`` allows PAM mismatches, ``"spacer"`` does not.
        generator: Optional torch generator.

    Returns:
        Long tensor ``[23]``: the decoy site in array order.
    """
    if guide.numel() != SITE_LEN:
        raise ValueError(f"guide must be {SITE_LEN} tokens, got {guide.numel()}")
    if mismatch_span not in MISMATCH_SPANS:
        raise ValueError(f"mismatch_span must be one of {MISMATCH_SPANS}, got {mismatch_span!r}")

    span = SITE_LEN if mismatch_span == "site" else SPACER_LEN
    n_mismatch = min(int(n_mismatch), span)
    decoy = guide.long().clone()
    if n_mismatch > 0:
        pos = _choose_without_replacement(
            n_mismatch, torch.ones(span, dtype=torch.double), generator
        )
        for i in pos:
            decoy[i] = _random_base(generator, exclude=int(decoy[i]))
    return decoy


# ---------------------------------------------------------------------------
# Leakage checks (spec section 6.2)
# ---------------------------------------------------------------------------

def structural_leakage_report(samples: Sequence[dict]) -> dict:
    """Check that nothing in a batch of samples reveals ``(b_D, b_R)`` by shape.

    This is the *hard* half of the leakage probe. A learned classifier can
    always do somewhat better than chance on bulge configuration by reading the
    sequences themselves -- that is the task, not a leak. What must be exactly
    invariant is the structure: guide length, window length, band width and the
    extent of the region label. If any of those tracks the bulge configuration,
    the model can read the answer off the input and every downstream result is
    void.

    Args:
        samples: Dicts with keys ``"guide"``, ``"window"``, ``"band"``,
            ``"edits"`` (``(m, b_D, b_R)``). Extra keys are ignored.

    Returns:
        Dict with ``"ok"`` (bool), the observed shape sets, and the observed
        bulge configurations.
    """
    guide_lens = {int(s["guide"].numel()) for s in samples}
    window_lens = {int(s["window"].numel()) for s in samples}
    band_lens = {int(s["band"].numel()) for s in samples}
    configs = sorted({(int(s["edits"][1]), int(s["edits"][2])) for s in samples})
    ok = len(guide_lens) == 1 and len(window_lens) == 1 and len(band_lens) == 1
    return {
        "ok": ok,
        "guide_lengths": sorted(guide_lens),
        "window_lengths": sorted(window_lens),
        "band_widths": sorted(band_lens),
        "bulge_configs": configs,
        "n_samples": len(samples),
    }
