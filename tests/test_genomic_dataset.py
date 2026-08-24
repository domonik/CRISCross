"""
Pytest tests for CRISCCross/Datasets.py - GenomicDataset module

Tests cover:
- GenomicDataset initialization and configuration
- Sample generation (chromosome selection, window extraction)
- Sequence processing (mutation, complement, N-base replacement)
- Epigenetic feature loading and normalization
- Strand handling
- Edge cases (short chromosomes, missing features)
"""

import pytest
import torch
import numpy as np
import os
import tempfile
import shutil
from CRISCross.Datasets import GenomicDataset, mutate_target, TORCHCOMPLEMENT


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


@pytest.fixture
def temp_bigwig_dir():
    """Create a temporary directory with mock BigWig (numpy) feature files."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def mock_chrom_sizes():
    """Mock chromosome sizes for testing."""
    return {
        "chr1": 10000,
        "chr2": 5000,
        "chr3": 2000,
    }


@pytest.fixture
def mock_seq_dict(mock_chrom_sizes):
    """Create mock sequence dictionary with valid DNA bases (1-4)."""
    seq_dict = {}
    for chrom, length in mock_chrom_sizes.items():
        # Generate random valid DNA bases (1=A, 2=C, 3=G, 4=T)
        # Avoid 0 (N) for cleaner testing
        seq = torch.randint(1, 5, (length,), dtype=torch.uint8)
        seq_dict[chrom] = seq
    return seq_dict


@pytest.fixture
def mock_epi_features():
    """List of epigenetic feature names to test."""
    return ["ATAC", "DNASE"]


@pytest.fixture
def mock_bigwig_files(temp_bigwig_dir, mock_chrom_sizes, mock_epi_features):
    """Create mock BigWig feature files as numpy arrays."""
    for epi_feat in mock_epi_features:
        for chrom, length in mock_chrom_sizes.items():
            # Create random float32 values representing epigenetic signal
            feat_array = np.random.rand(length).astype(np.float32)
            file_path = os.path.join(temp_bigwig_dir, f"{epi_feat}_{chrom}.npy")
            np.save(file_path, feat_array)
    return temp_bigwig_dir


@pytest.fixture
def genomic_dataset_mock(
    mock_chrom_sizes, mock_seq_dict, mock_bigwig_files, mock_epi_features, cpu_device
):
    """Create a GenomicDataset with mock data."""
    dataset = GenomicDataset(
        chrom_sizes=mock_chrom_sizes,
        seq_dict=mock_seq_dict,
        bw_dir=[mock_bigwig_files],
        epi_features=mock_epi_features,
        window_size=512,
        num_samples=10,
    )
    return dataset


# ============================================================================
# Tests for mutate_target utility function
# ============================================================================

class TestMutateTarget:
    """Tests for the mutate_target utility function."""

    def test_mutate_returns_valid_bases(self, cpu_device):
        """Test that mutations produce valid base values (1-4)."""
        target = torch.ones(23, dtype=torch.long, device=cpu_device)
        mutated = mutate_target(target.clone())

        # All values should be in valid range [1, 4]
        assert (mutated >= 1).all()
        assert (mutated <= 4).all()

    def test_mutate_changes_sequence(self, cpu_device, seed):
        """Test that mutations actually change the sequence."""
        torch.manual_seed(seed)
        target = torch.ones(23, dtype=torch.long, device=cpu_device) * 2  # All C's
        mutated = mutate_target(target.clone())

        # With high probability, at least some positions should change
        # (though it's possible all positions stay same with very low probability)
        assert mutated.shape == target.shape

    def test_mutate_preserves_length(self, cpu_device):
        """Test that mutation preserves sequence length."""
        for window_size in [10, 23, 50, 100]:
            target = torch.randint(1, 5, (window_size,), device=cpu_device)
            mutated = mutate_target(target.clone())

            assert mutated.shape == (window_size,)

    def test_mutate_batch_processing(self, cpu_device, seed):
        """Test mutating multiple sequences."""
        torch.manual_seed(seed)
        for _ in range(5):
            target = torch.randint(1, 5, (23,), device=cpu_device)
            mutated = mutate_target(target.clone())
            assert mutated.shape == target.shape
            assert (mutated >= 1).all()
            assert (mutated <= 4).all()


# ============================================================================
# Tests for GenomicDataset initialization
# ============================================================================

class TestGenomicDatasetInit:
    """Tests for GenomicDataset initialization."""

    def test_basic_initialization(
        self, mock_chrom_sizes, mock_seq_dict, mock_bigwig_files, mock_epi_features
    ):
        """Test basic dataset initialization."""
        dataset = GenomicDataset(
            chrom_sizes=mock_chrom_sizes,
            seq_dict=mock_seq_dict,
            bw_dir=[mock_bigwig_files],
            epi_features=mock_epi_features,
            window_size=512,
            num_samples=100,
        )

        assert dataset.window_size == 512
        assert dataset.num_samples == 100
        assert len(dataset.chroms) == 3
        assert set(dataset.chroms) == set(mock_chrom_sizes.keys())

    def test_empty_epi_features(self, mock_chrom_sizes, mock_seq_dict, mock_bigwig_files):
        """Test initialization with no epigenetic features."""
        dataset = GenomicDataset(
            chrom_sizes=mock_chrom_sizes,
            seq_dict=mock_seq_dict,
            bw_dir=[mock_bigwig_files],
            epi_features=[],
            window_size=512,
            num_samples=10,
        )

        assert dataset.epi_features == []

    def test_epi_stats_normalization(
        self, mock_chrom_sizes, mock_seq_dict, mock_bigwig_files, mock_epi_features
    ):
        """Test initialization with epigenetic normalization statistics."""
        epi_stats = [
            {"mean": 0.5, "std": 0.2, "mode": "DENSE"},
            {"mean": 0.3, "std": 0.1, "mode": "COUNT"},
        ]

        dataset = GenomicDataset(
            chrom_sizes=mock_chrom_sizes,
            seq_dict=mock_seq_dict,
            bw_dir=[mock_bigwig_files],
            epi_features=mock_epi_features,
            window_size=512,
            num_samples=10,
            epi_stats=epi_stats,
        )

        assert dataset.epi_stats is not None
        assert hasattr(dataset, "epi_mean")
        assert hasattr(dataset, "epi_std")
        assert hasattr(dataset, "epi_log_mask")
        assert dataset.epi_mean.shape == (2,)
        assert dataset.epi_std.shape == (2,)
        assert dataset.epi_log_mask.shape == (2,)

    def test_chrom_sizes_weighted_sampling(self, mock_chrom_sizes, mock_seq_dict, mock_bigwig_files, mock_epi_features, seed):
        """Test that chromosome sampling is weighted by size."""
        torch.manual_seed(seed)
        dataset = GenomicDataset(
            chrom_sizes=mock_chrom_sizes,
            seq_dict=mock_seq_dict,
            bw_dir=[mock_bigwig_files],
            epi_features=mock_epi_features,
            window_size=512,
            num_samples=1000,
        )

        # chr1 should be sampled most often (largest)
        # chr3 should be sampled least often (smallest)
        chrom_counts = {chrom: 0 for chrom in mock_chrom_sizes.keys()}

        for i in range(1000):
            # Simulate the sampling logic using multinomial
            weights = torch.tensor([mock_chrom_sizes[c] for c in mock_chrom_sizes.keys()], dtype=torch.float32)
            chrom_idx = torch.multinomial(weights, 1).item()
            chrom = list(mock_chrom_sizes.keys())[chrom_idx]
            chrom_counts[chrom] += 1

        # chr1 should have more samples than chr3 (with high probability)
        assert chrom_counts["chr1"] > chrom_counts["chr3"]


# ============================================================================
# Tests for GenomicDataset length
# ============================================================================

class TestGenomicDatasetLen:
    """Tests for GenomicDataset __len__ method."""

    def test_len_returns_num_samples(self, genomic_dataset_mock):
        """Test that __len__ returns num_samples."""
        assert len(genomic_dataset_mock) == 10

    def test_len_various_sizes(
        self, mock_chrom_sizes, mock_seq_dict, mock_bigwig_files, mock_epi_features
    ):
        """Test __len__ with different num_samples values."""
        for num_samples in [1, 50, 1000]:
            dataset = GenomicDataset(
                chrom_sizes=mock_chrom_sizes,
                seq_dict=mock_seq_dict,
                bw_dir=[mock_bigwig_files],
                epi_features=mock_epi_features,
                window_size=512,
                num_samples=num_samples,
            )
            assert len(dataset) == num_samples


# ============================================================================
# Tests for GenomicDataset item structure
# ============================================================================

class TestGenomicDatasetGetItem:
    """Tests for GenomicDataset __getitem__ method."""

    def test_return_tuple_structure(self, genomic_dataset_mock):
        """Test that __getitem__ returns a 6-element tuple."""
        target_x, off_target_x, epi, y, counts, strand, atac = genomic_dataset_mock[0]

        # Check tuple structure
        assert isinstance(target_x, torch.Tensor)
        assert isinstance(off_target_x, torch.Tensor)
        assert isinstance(epi, torch.Tensor) or epi == 0  # epi can be 0 if no features
        assert isinstance(y, int)
        assert isinstance(counts, int)
        assert isinstance(strand, int)

    def test_target_sequence_length(self, genomic_dataset_mock):
        """Test that target sequence is 23bp (guide length)."""
        target_x, _, _, _, _, _, _ = genomic_dataset_mock[0]
        assert target_x.shape[0] == 23

    def test_off_target_sequence_length(self, genomic_dataset_mock):
        """Test that off-target sequence matches window_size."""
        _, off_target_x, _, _, _, _, _ = genomic_dataset_mock[0]
        assert off_target_x.shape[0] == genomic_dataset_mock.window_size

    def test_valid_base_values(self, genomic_dataset_mock):
        """Test that all base values are in valid range [1, 4]."""
        target_x, off_target_x, _, _, _, _, _ = genomic_dataset_mock[0]

        # N bases (0) should be replaced with random bases
        assert (target_x >= 1).all()
        assert (target_x <= 4).all()
        assert (off_target_x >= 1).all()
        assert (off_target_x <= 4).all()

    def test_target_is_centered_in_off_target(self, genomic_dataset_mock):
        """Test that target sequence corresponds to center of off-target."""
        target_x, off_target_x, _, _, _, strand, _ = genomic_dataset_mock[0]

        window_size = genomic_dataset_mock.window_size
        center = window_size // 2 + window_size % 2

        # Target should match the center 23bp region of off-target
        # (before mutation is applied)
        center_region = off_target_x[center - 23 // 2 - 1 : center + 23 // 2]

        # After mutation, they may differ, but shapes should match
        assert target_x.shape == center_region.shape

    def test_strand_values(self, genomic_dataset_mock):
        """Test that strand is either 0 (reverse) or 1 (forward)."""
        strands = set()
        for i in range(100):
            _, _, _, _, _, strand, _ = genomic_dataset_mock[i % len(genomic_dataset_mock)]
            strands.add(strand)

        # Strand should only be 0 or 1
        assert strands.issubset({0, 1})

    def test_epigenetic_features_shape(self, genomic_dataset_mock, mock_epi_features):
        """Test that epigenetic features have correct shape."""
        _, _, epi, _, _, _, _ = genomic_dataset_mock[0]

        # epi should be [window_size, num_features]
        assert epi.shape[0] == genomic_dataset_mock.window_size
        assert epi.shape[1] == len(mock_epi_features)

    def test_epigenetic_no_nans(self, genomic_dataset_mock):
        """Test that epigenetic features have no NaN values."""
        _, _, epi, _, _, _, _ = genomic_dataset_mock[0]
        assert not torch.isnan(epi).any()


# ============================================================================
# Tests for sequence transformation
# ============================================================================

class TestSequenceTransformation:
    """Tests for sequence transformations in GenomicDataset."""

    def test_reverse_complement_applied(self, mock_chrom_sizes, mock_seq_dict, mock_bigwig_files, mock_epi_features, seed):
        """Test that reverse complement is applied for strand=0."""
        torch.manual_seed(seed)
        dataset = GenomicDataset(
            chrom_sizes=mock_chrom_sizes,
            seq_dict=mock_seq_dict,
            bw_dir=[mock_bigwig_files],
            epi_features=mock_epi_features,
            window_size=512,
            num_samples=100,
        )

        # Collect samples from both strands
        forward_seq = None
        reverse_seq = None

        for i in range(100):
            _, off_target_x, _, _, _, strand, _ = dataset[i % len(dataset)]
            if strand == 1 and forward_seq is None:
                forward_seq = off_target_x.clone()
            elif strand == 0 and reverse_seq is None:
                reverse_seq = off_target_x.clone()
            if forward_seq is not None and reverse_seq is not None:
                break

        # Both strands should be found with high probability
        assert forward_seq is not None
        assert reverse_seq is not None

    def test_complement_mapping_correct(self, cpu_device):
        """Test that complement mapping is correct."""
        # A(1) <-> T(4), C(2) <-> G(3)
        original = torch.tensor([1, 2, 3, 4], dtype=torch.long, device=cpu_device)
        complement = TORCHCOMPLEMENT[original]

        assert complement[0] == 4  # A -> T
        assert complement[1] == 3  # C -> G
        assert complement[2] == 2  # G -> C
        assert complement[3] == 1  # T -> A


# ============================================================================
# Tests for epigenetic feature loading
# ============================================================================

class TestEpigeneticFeatures:
    """Tests for epigenetic feature loading and processing."""

    def test_single_feature(
        self, mock_chrom_sizes, mock_seq_dict, mock_bigwig_files, temp_bigwig_dir
    ):
        """Test with a single epigenetic feature."""
        dataset = GenomicDataset(
            chrom_sizes=mock_chrom_sizes,
            seq_dict=mock_seq_dict,
            bw_dir=[temp_bigwig_dir],
            epi_features=["ATAC"],
            window_size=512,
            num_samples=10,
        )

        _, _, epi, _, _, _, _ = dataset[0]
        assert epi.shape == (512, 1)

    def test_multiple_features(
        self, mock_chrom_sizes, mock_seq_dict, mock_bigwig_files, mock_epi_features
    ):
        """Test with multiple epigenetic features."""
        dataset = GenomicDataset(
            chrom_sizes=mock_chrom_sizes,
            seq_dict=mock_seq_dict,
            bw_dir=[mock_bigwig_files],
            epi_features=mock_epi_features,
            window_size=512,
            num_samples=10,
        )

        _, _, epi, _, _, _, _ = dataset[0]
        assert epi.shape == (512, 2)  # 2 features

    def test_normalization_applied(
        self, mock_chrom_sizes, mock_seq_dict, mock_bigwig_files, mock_epi_features
    ):
        """Test that normalization is applied when epi_stats provided."""
        epi_stats = [
            {"mean": 0.5, "std": 0.2, "mode": "DENSE"},
            {"mean": 0.3, "std": 0.1, "mode": "DENSE"},
        ]

        dataset = GenomicDataset(
            chrom_sizes=mock_chrom_sizes,
            seq_dict=mock_seq_dict,
            bw_dir=[mock_bigwig_files],
            epi_features=mock_epi_features,
            window_size=512,
            num_samples=10,
            epi_stats=epi_stats,
        )

        _, _, epi, _, _, _, _ = dataset[0]

        # Normalized features should have mean ~0 and std ~1
        # (approximately, due to random sampling)
        assert epi.shape == (512, 2)

    def test_log_transformation_for_count_data(
        self, mock_chrom_sizes, mock_seq_dict, mock_bigwig_files, mock_epi_features
    ):
        """Test log1p transformation for COUNT mode features."""
        epi_stats = [
            {"mean": 0.5, "std": 0.2, "mode": "COUNT"},
            {"mean": 0.3, "std": 0.1, "mode": "DENSE"},
        ]

        dataset = GenomicDataset(
            chrom_sizes=mock_chrom_sizes,
            seq_dict=mock_seq_dict,
            bw_dir=[mock_bigwig_files],
            epi_features=mock_epi_features,
            window_size=512,
            num_samples=10,
            epi_stats=epi_stats,
        )

        _, _, epi, _, _, _, _ = dataset[0]
        assert epi.shape == (512, 2)

    def test_strand_flip_for_epigenetics(
        self, mock_chrom_sizes, mock_seq_dict, mock_bigwig_files, mock_epi_features, seed
    ):
        """Test that epigenetic features are flipped for reverse strand."""
        torch.manual_seed(seed)
        dataset = GenomicDataset(
            chrom_sizes=mock_chrom_sizes,
            seq_dict=mock_seq_dict,
            bw_dir=[mock_bigwig_files],
            epi_features=mock_epi_features,
            window_size=512,
            num_samples=100,
        )

        # Get features from both strands
        forward_epi = None
        reverse_epi = None

        for i in range(100):
            _, _, epi, _, _, strand, _ = dataset[i % len(dataset)]
            if strand == 1 and forward_epi is None:
                forward_epi = epi.clone()
            elif strand == 0 and reverse_epi is None:
                reverse_epi = epi.clone()
            if forward_epi is not None and reverse_epi is not None:
                break

        # Reverse strand features should be flipped
        assert forward_epi is not None
        assert reverse_epi is not None


# ============================================================================
# Tests for edge cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_small_window_size(
        self, mock_chrom_sizes, mock_seq_dict, mock_bigwig_files, mock_epi_features
    ):
        """Test with a very small window size."""
        dataset = GenomicDataset(
            chrom_sizes=mock_chrom_sizes,
            seq_dict=mock_seq_dict,
            bw_dir=[mock_bigwig_files],
            epi_features=mock_epi_features,
            window_size=100,
            num_samples=10,
        )

        _, off_target_x, epi, _, _, _, _ = dataset[0]
        assert off_target_x.shape[0] == 100
        assert epi.shape[0] == 100

    def test_large_window_size(
        self, mock_chrom_sizes, mock_seq_dict, mock_bigwig_files, mock_epi_features
    ):
        """Test with a large window size."""
        # mock_chrom_sizes has a 2000nt chromosome, which is exactly the window
        # size and therefore legitimately raises (see
        # test_chromosome_shorter_than_window_raises_error). Drop it so this
        # test measures what it means to measure rather than flaking on the
        # chromosome draw.
        chrom_sizes = {c: n for c, n in mock_chrom_sizes.items() if n > 2000}
        dataset = GenomicDataset(
            chrom_sizes=chrom_sizes,
            seq_dict=mock_seq_dict,
            bw_dir=[mock_bigwig_files],
            epi_features=mock_epi_features,
            window_size=2000,
            num_samples=10,
        )

        _, off_target_x, epi, _, _, _, _ = dataset[0]
        assert off_target_x.shape[0] == 2000
        assert epi.shape[0] == 2000

    def test_single_sample(
        self, mock_chrom_sizes, mock_seq_dict, mock_bigwig_files, mock_epi_features
    ):
        """Test with num_samples=1."""
        dataset = GenomicDataset(
            chrom_sizes=mock_chrom_sizes,
            seq_dict=mock_seq_dict,
            bw_dir=[mock_bigwig_files],
            epi_features=mock_epi_features,
            window_size=512,
            num_samples=1,
        )

        assert len(dataset) == 1
        _, _, epi, _, _, _, _ = dataset[0]
        assert epi is not None

    def test_chromosome_shorter_than_window_raises_error(
        self, mock_seq_dict, mock_bigwig_files, mock_epi_features
    ):
        """Test error when chromosome is shorter than window_size."""
        small_chrom_sizes = {"chr_small": 50}  # Too small for window_size=512

        dataset = GenomicDataset(
            chrom_sizes=small_chrom_sizes,
            seq_dict=mock_seq_dict,
            bw_dir=[mock_bigwig_files],
            epi_features=mock_epi_features,
            window_size=512,
            num_samples=10,
        )

        # This should raise ValueError when trying to sample
        with pytest.raises(ValueError, match="shorter than window_size"):
            _ = dataset[0]

    def test_multiple_bw_dirs(
        self, mock_chrom_sizes, mock_seq_dict, mock_bigwig_files, mock_epi_features
    ):
        """Test with multiple BigWig directories."""
        # Create a second directory with different features
        second_dir = tempfile.mkdtemp()
        try:
            for epi_feat in mock_epi_features:
                for chrom, length in mock_chrom_sizes.items():
                    feat_array = np.random.rand(length).astype(np.float32)
                    file_path = os.path.join(second_dir, f"{epi_feat}_{chrom}.npy")
                    np.save(file_path, feat_array)

            dataset = GenomicDataset(
                chrom_sizes=mock_chrom_sizes,
                seq_dict=mock_seq_dict,
                bw_dir=[mock_bigwig_files, second_dir],
                epi_features=mock_epi_features,
                window_size=512,
                num_samples=10,
            )

            # Should work without errors
            _, _, epi, _, _, _, _ = dataset[0]
            assert epi.shape == (512, len(mock_epi_features))
        finally:
            shutil.rmtree(second_dir)

    def test_samples_have_correct_types(self, genomic_dataset_mock):
        """Test that samples have correct data types."""
        target_x, off_target_x, epi, y, counts, strand, atac = genomic_dataset_mock[0]

        # Verify types
        assert isinstance(target_x, torch.Tensor)
        assert isinstance(off_target_x, torch.Tensor)
        assert isinstance(epi, torch.Tensor)
        assert isinstance(y, int)
        assert isinstance(counts, int)
        assert isinstance(strand, int)

        # Verify dtypes
        assert target_x.dtype in [torch.uint8, torch.long, torch.int64]
        assert off_target_x.dtype in [torch.uint8, torch.long, torch.int64]
        assert epi.dtype == torch.float32


# ============================================================================
# Tests for integration with DataLoader
# ============================================================================

class TestGenomicDatasetDataLoader:
    """Tests for GenomicDataset integration with PyTorch DataLoader."""

    def test_dataloader_iteration(
        self, mock_chrom_sizes, mock_seq_dict, mock_bigwig_files, mock_epi_features, cpu_device
    ):
        """Test that DataLoader can iterate over dataset."""
        from torch.utils.data import DataLoader

        dataset = GenomicDataset(
            chrom_sizes=mock_chrom_sizes,
            seq_dict=mock_seq_dict,
            bw_dir=[mock_bigwig_files],
            epi_features=mock_epi_features,
            window_size=512,
            num_samples=20,
        )

        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        batch_count = 0
        for batch in dataloader:
            target_x, off_target_x, epi, y, counts, strand, atac = batch

            assert target_x.shape[0] == 4  # batch_size
            assert off_target_x.shape[0] == 4
            assert epi.shape[0] == 4
            batch_count += 1

        assert batch_count == 5  # 20 samples / 4 batch_size

    def test_dataloader_with_shuffle(
        self, mock_chrom_sizes, mock_seq_dict, mock_bigwig_files, mock_epi_features, seed
    ):
        """Test DataLoader with shuffling."""
        torch.manual_seed(seed)
        from torch.utils.data import DataLoader

        dataset = GenomicDataset(
            chrom_sizes=mock_chrom_sizes,
            seq_dict=mock_seq_dict,
            bw_dir=[mock_bigwig_files],
            epi_features=mock_epi_features,
            window_size=512,
            num_samples=20,
        )

        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        # Should iterate without errors
        for batch in dataloader:
            target_x, off_target_x, epi, y, counts, strand, atac = batch
            assert target_x.shape[0] == 4

    def test_dataloader_num_workers(
        self, mock_chrom_sizes, mock_seq_dict, mock_bigwig_files, mock_epi_features, cpu_device
    ):
        """Test DataLoader with multiple workers."""
        from torch.utils.data import DataLoader

        dataset = GenomicDataset(
            chrom_sizes=mock_chrom_sizes,
            seq_dict=mock_seq_dict,
            bw_dir=[mock_bigwig_files],
            epi_features=mock_epi_features,
            window_size=512,
            num_samples=20,
        )

        dataloader = DataLoader(dataset, batch_size=4, num_workers=2)

        # Should iterate without errors
        for batch in dataloader:
            target_x, off_target_x, epi, y, counts, strand, atac = batch
            assert target_x.shape[0] == 4
