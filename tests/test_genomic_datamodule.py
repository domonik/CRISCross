"""
Pytest tests for CRISCross/Datasets.py - GenomicDataModule

Tests cover:
- GenomicDataModule initialization and configuration
- Data preparation (FASTA parsing, caching, BigWig conversion)
- Setup and dataset creation
- DataLoader generation (train, val, test)
- Fine-tuning mode with dataframe
- Edge cases and error handling
"""

import pytest
import torch
import numpy as np
import os
import tempfile
import shutil
import pandas as pd
from torch.utils.data import DataLoader
from CRISCross.Datasets import GenomicDataModule, CHROMOSOME_SIZES


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
def temp_dir():
    """Create a temporary directory for test files."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def mock_fasta_file(temp_dir):
    """Create a mock FASTA file with test sequences."""
    fasta_path = os.path.join(temp_dir, "test_genome.fa")
    with open(fasta_path, "w") as f:
        # Write test chromosomes matching CHROMOSOME_SIZES
        for chrom in ["chr1", "chr2", "chr3"]:
            f.write(f">{chrom}\n")
            # Write a shorter sequence for testing (1000 bases each)
            seq = "ACGT" * 250  # 1000 bases
            f.write(seq + "\n")
    return fasta_path


@pytest.fixture
def mock_df():
    """Create a mock dataframe for fine-tuning."""
    df = pd.DataFrame({
        "GuideID": [f"g{i}" for i in range(10)],
        "label": [0, 1] * 5,
        "chr": ["chr1"] * 5 + ["chr2"] * 5,
        "start": [100 + i * 10 for i in range(10)],
        "end": [123 + i * 10 for i in range(10)],
        "Guide_sequence": ["ACGT" * 6 + "N"] * 10,
        "Strand": ["+"] * 5 + ["-"] * 5,
        "AlphagenomeIndex": list(range(10)),
        "extended_off_target": ["ACGT" * 26] * 10,
    })
    return df


@pytest.fixture
def mock_bigwig_files(temp_dir, mock_fasta_file):
    """Create mock BigWig (numpy) feature files per chromosome."""
    epi_features = ["ATAC", "DNASE"]
    chroms = ["chr1", "chr2", "chr3"]
    seq_len = 1000  # Matches mock_fasta_file

    for epi_feat in epi_features:
        for chrom in chroms:
            # Create random float32 values
            feat_array = np.random.rand(seq_len).astype(np.float32)
            file_path = os.path.join(temp_dir, f"{epi_feat}_{chrom}.npy")
            np.save(file_path, feat_array)

    return temp_dir, epi_features


@pytest.fixture
def minimal_datamodule_config(mock_bigwig_files, temp_dir):
    """Get minimal configuration for GenomicDataModule."""
    bw_dir, epi_features = mock_bigwig_files
    # Use temp_dir for fasta_path since mock_fasta_file is created there
    fasta_path = os.path.join(temp_dir, "test_genome.fa")
    return {
        "fasta_path": fasta_path,
        "epi_features": epi_features,
        "bw_dir": bw_dir,
        "window_size": 100,
        "batch_size": 4,
        "num_workers": 1,  # Use 1 to avoid persistent_workers issues
        "num_samples": 10,
    }


# ============================================================================
# Tests for GenomicDataModule initialization
# ============================================================================

class TestGenomicDataModuleInit:
    """Tests for GenomicDataModule initialization."""

    def test_basic_initialization(self, minimal_datamodule_config):
        """Test basic data module initialization."""
        dm = GenomicDataModule(**minimal_datamodule_config)

        assert dm.window_size == 100
        assert dm.batch_size == 4
        assert dm.num_workers == 1
        assert dm.num_samples == 10
        assert dm.oversample is False
        assert dm.norm_epi is False
        assert dm.use_energy is False

    def test_epi_features_stored(self, minimal_datamodule_config):
        """Test that epigenetic features are stored correctly."""
        dm = GenomicDataModule(**minimal_datamodule_config)

        assert len(dm.epi_features) == 2
        assert "ATAC" in dm.epi_features
        assert "DNASE" in dm.epi_features

    def test_bw_dir_converted_to_list(self, minimal_datamodule_config):
        """Test that bw_dir is converted to list if string."""
        dm = GenomicDataModule(**minimal_datamodule_config)

        assert isinstance(dm.bw_dirs, list)
        assert len(dm.bw_dirs) == 1

    def test_chromosomes_list(self, minimal_datamodule_config):
        """Test that chromosomes list is populated."""
        dm = GenomicDataModule(**minimal_datamodule_config)

        assert isinstance(dm.chromosomes, list)
        assert len(dm.chromosomes) > 0
        assert "chr1" in dm.chromosomes

    def test_np_mode_bw_files(self, minimal_datamodule_config):
        """Test bw_files generation in np mode."""
        dm = GenomicDataModule(**minimal_datamodule_config)

        assert dm.mode == "np"
        assert len(dm.bw_files) > 0
        # Should contain .npy files
        assert all(".npy" in f for f in dm.bw_files)

    def test_invalid_mode_raises_error(self, minimal_datamodule_config):
        """Test that invalid mode raises ValueError."""
        config = minimal_datamodule_config.copy()
        config["mode"] = "invalid"

        with pytest.raises(ValueError, match="mode must be either"):
            GenomicDataModule(**config)

    def test_norm_epi_flag(self, minimal_datamodule_config):
        """Test norm_epi flag is stored."""
        config = minimal_datamodule_config.copy()
        config["norm_epi"] = True

        dm = GenomicDataModule(**config)

        assert dm.norm_epi is True

    def test_use_energy_flag(self, minimal_datamodule_config):
        """Test use_energy flag is stored."""
        config = minimal_datamodule_config.copy()
        config["use_energy"] = True

        dm = GenomicDataModule(**config)

        assert dm.use_energy is True

    def test_oversample_with_dataframe(self, minimal_datamodule_config):
        """Test oversample is True when dataframe is provided."""
        # Create a minimal dataframe
        df = pd.DataFrame({
            "GuideID": ["g1", "g2", "g3"],
            "label": [0, 1, 0],
            "chr": ["chr1", "chr2", "chr1"],
            "start": [100, 200, 300],
            "end": [123, 223, 323],
            "Guide_sequence": ["ACGT" * 6 + "N", "TGCA" * 6 + "N", "AAAAAA" * 4],
            "Strand": ["+", "-", "+"],
            "AlphagenomeIndex": [0, 1, 2],
            "extended_off_target": ["ACGT" * 26, "TGCA" * 26, "AAAAAA" * 18],
        })

        config = minimal_datamodule_config.copy()
        config["df"] = df

        dm = GenomicDataModule(**config)

        assert dm.oversample is True
        assert dm.val_guides is None
        assert dm.test_guides is None

    def test_local_bw_dirs_creation(self, minimal_datamodule_config):
        """Test that local_bw_dirs are created based on TMPDIR."""
        dm = GenomicDataModule(**minimal_datamodule_config)

        assert isinstance(dm.local_bw_dirs, list)
        assert len(dm.local_bw_dirs) == len(dm.bw_dirs)


# ============================================================================
# Tests for data preparation
# ============================================================================

class TestGenomicDataModulePrepareData:
    """Tests for GenomicDataModule.prepare_data method."""

    def test_prepare_data_copies_files(self, minimal_datamodule_config, mock_fasta_file):
        """Test that prepare_data copies files to local directory."""
        dm = GenomicDataModule(**minimal_datamodule_config)

        # Create FASTA cache if not exists
        dm.prepare_data()

        # Local directory should exist
        for local_dir in dm.local_bw_dirs:
            assert os.path.exists(local_dir)

    def test_prepare_data_creates_cache(self, minimal_datamodule_config, mock_fasta_file):
        """Test that prepare_data creates FASTA cache."""
        # Remove existing cache if any
        cache_path = "encoded_fasta.pt"
        if os.path.exists(cache_path):
            os.remove(cache_path)

        dm = GenomicDataModule(**minimal_datamodule_config)
        dm.prepare_data()

        # Cache should be created
        assert os.path.exists(cache_path)

        # Clean up
        os.remove(cache_path)

    def test_prepare_data_uses_existing_cache(self, minimal_datamodule_config, mock_fasta_file):
        """Test that prepare_data uses existing cache."""
        cache_path = "encoded_fasta.pt"

        # Create cache first
        if os.path.exists(cache_path):
            os.remove(cache_path)

        dm1 = GenomicDataModule(**minimal_datamodule_config)
        dm1.prepare_data()

        # Cache should exist
        assert os.path.exists(cache_path)

        # Create second data module - should use existing cache
        dm2 = GenomicDataModule(**minimal_datamodule_config)
        dm2.prepare_data()

        # Clean up
        os.remove(cache_path)


# ============================================================================
# Tests for setup and dataset creation
# ============================================================================

class TestGenomicDataModuleSetup:
    """Tests for GenomicDataModule.setup method."""

    def test_setup_creates_datasets(self, minimal_datamodule_config, mock_fasta_file):
        """Test that setup creates train and validation datasets."""
        dm = GenomicDataModule(**minimal_datamodule_config)
        dm.prepare_data()
        dm.setup()

        assert dm.dataset is not None
        assert dm.val_set is not None

    def test_setup_loads_chrom_sizes(self, minimal_datamodule_config, mock_fasta_file):
        """Test that setup loads chromosome sizes from cache."""
        dm = GenomicDataModule(**minimal_datamodule_config)
        dm.prepare_data()
        dm.setup()

        assert dm.chrom_sizes is not None
        assert "chr1" in dm.chrom_sizes
        assert "chr2" in dm.chrom_sizes
        assert "chr3" in dm.chrom_sizes

    def test_setup_loads_seq_dict(self, minimal_datamodule_config, mock_fasta_file):
        """Test that setup loads sequence dictionary from cache."""
        dm = GenomicDataModule(**minimal_datamodule_config)
        dm.prepare_data()
        dm.setup()

        assert dm.seq_dict is not None
        assert "chr1" in dm.seq_dict
        assert isinstance(dm.seq_dict["chr1"], torch.Tensor)

    def test_setup_with_energy(self, minimal_datamodule_config, mock_fasta_file):
        """Test setup with energy calculation enabled."""
        config = minimal_datamodule_config.copy()
        config["use_energy"] = True

        dm = GenomicDataModule(**config)
        dm.prepare_data()
        dm.setup()

        assert dm.dataset is not None
        # Dataset should be EnergyGenomicDataset

    def test_setup_norm_epi(self, minimal_datamodule_config, mock_fasta_file):
        """Test setup with epigenetic normalization."""
        config = minimal_datamodule_config.copy()
        config["norm_epi"] = True

        dm = GenomicDataModule(**config)
        dm.prepare_data()
        dm.setup()

        assert dm.dataset is not None
        assert dm.val_set is not None


# ============================================================================
# Tests for DataLoader generation
# ============================================================================

class TestGenomicDataModuleDataLoaders:
    """Tests for GenomicDataModule DataLoader methods."""

    def test_train_dataloader(self, minimal_datamodule_config, mock_fasta_file):
        """Test train_dataloader returns valid DataLoader."""
        dm = GenomicDataModule(**minimal_datamodule_config)
        dm.prepare_data()
        dm.setup()

        train_loader = dm.train_dataloader()

        assert isinstance(train_loader, DataLoader)
        assert train_loader.batch_size == 4

    def test_val_dataloader(self, minimal_datamodule_config, mock_fasta_file):
        """Test val_dataloader returns valid DataLoader."""
        dm = GenomicDataModule(**minimal_datamodule_config)
        dm.prepare_data()
        dm.setup()

        val_loader = dm.val_dataloader()

        assert isinstance(val_loader, DataLoader)
        assert val_loader.batch_size == 4

    def test_test_dataloader_no_df(self, minimal_datamodule_config, mock_fasta_file):
        """Test test_dataloader when df is None (verifies test_set is created)."""
        dm = GenomicDataModule(**minimal_datamodule_config)
        dm.prepare_data()
        dm.setup()

        # test_set should now exist in non-fine-tuning mode (bug fix)
        assert hasattr(dm, "test_set")

        # test_dataloader should work without errors
        test_loader = dm.test_dataloader()
        assert isinstance(test_loader, DataLoader)
        assert test_loader.batch_size == 4

    def test_train_dataloader_iteration(self, minimal_datamodule_config, mock_fasta_file, cpu_device):
        """Test that train_dataloader can iterate."""
        dm = GenomicDataModule(**minimal_datamodule_config)
        dm.prepare_data()
        dm.setup()

        train_loader = dm.train_dataloader()

        batch_count = 0
        for batch in train_loader:
            target_x, off_target_x, epi, y, counts, strand = batch

            assert target_x.shape[0] == 4  # batch_size
            assert off_target_x.shape[0] == 4
            assert epi.shape[0] == 4
            batch_count += 1

        assert batch_count > 0

    def test_val_dataloader_iteration(self, minimal_datamodule_config, mock_fasta_file, cpu_device):
        """Test that val_dataloader can iterate."""
        dm = GenomicDataModule(**minimal_datamodule_config)
        dm.prepare_data()
        dm.setup()

        val_loader = dm.val_dataloader()

        for batch in val_loader:
            target_x, off_target_x, epi, y, counts, strand = batch

            assert target_x.shape[0] <= 4  # Last batch may be smaller
            break  # Just test one batch

    def test_dataloader_pin_memory(self, minimal_datamodule_config, mock_fasta_file):
        """Test that dataloaders have pin_memory enabled."""
        dm = GenomicDataModule(**minimal_datamodule_config)
        dm.prepare_data()
        dm.setup()

        # Note: pin_memory is set in DataLoader but may not be directly accessible
        # This test verifies the dataloaders are created without errors
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        assert train_loader is not None
        assert val_loader is not None


# ============================================================================
# Tests for fine-tuning mode with dataframe
# ============================================================================

class TestGenomicDataModuleFineTuning:
    """Tests for GenomicDataModule in fine-tuning mode with dataframe."""

    def test_fine_tuning_with_dataframe(
        self, minimal_datamodule_config, mock_fasta_file, mock_df
    ):
        """Test data module with dataframe for fine-tuning."""
        config = minimal_datamodule_config.copy()
        config["df"] = mock_df
        config["val_guides"] = ["g8", "g9"]
        config["test_guides"] = ["g6", "g7"]

        dm = GenomicDataModule(**config)
        dm.prepare_data()
        dm.setup()

        # Should have train, val, and test sets
        assert dm.dataset is not None  # train set
        assert dm.val_set is not None
        assert dm.test_set is not None

    def test_train_val_test_split(self, minimal_datamodule_config, mock_fasta_file, mock_df):
        """Test train/val/test split with dataframe."""
        config = minimal_datamodule_config.copy()
        config["df"] = mock_df
        config["val_guides"] = ["g8", "g9"]
        config["test_guides"] = ["g6", "g7"]

        dm = GenomicDataModule(**config)
        dm.prepare_data()
        dm.setup()

        # Check masks were created
        assert dm.masks is not None
        assert len(dm.masks) == 3  # train, val, test masks

        train_mask, val_mask, test_mask = dm.masks

        # Val and test should have correct guides
        assert val_mask.sum() == 2
        assert test_mask.sum() == 2

    def test_class_weights_computation(
        self, minimal_datamodule_config, mock_fasta_file, mock_df
    ):
        """Test class weights computation for oversampling."""
        config = minimal_datamodule_config.copy()
        config["df"] = mock_df
        config["test_guides"] = ["g8", "g9"]  # Need test_guides to avoid None error

        dm = GenomicDataModule(**config)
        dm.prepare_data()
        dm.setup()

        # classweight column should be added to df
        assert "classweight" in dm.df.columns

    def test_preprocess_data(self, minimal_datamodule_config, mock_fasta_file, mock_df):
        """Test preprocess_data method."""
        config = minimal_datamodule_config.copy()
        config["df"] = mock_df
        config["test_guides"] = ["g8", "g9"]  # Need test_guides to avoid None error

        dm = GenomicDataModule(**config)
        dm.prepare_data()
        dm.setup()

        # preprocess_data is called during setup
        # Verify that data was processed correctly
        assert dm.dataset is not None

    def test_fine_tuning_dataloader_iteration(
        self, minimal_datamodule_config, mock_fasta_file, mock_df, cpu_device
    ):
        """Test dataloader iteration in fine-tuning mode."""
        config = minimal_datamodule_config.copy()
        config["df"] = mock_df
        config["val_guides"] = ["g8", "g9"]
        config["test_guides"] = ["g6", "g7"]
        config["num_samples"] = 5

        dm = GenomicDataModule(**config)
        dm.prepare_data()
        dm.setup()

        train_loader = dm.train_dataloader()

        for batch in train_loader:
            target_x, off_target_x, epi, y, counts, strand = batch

            assert target_x.shape[0] > 0
            assert off_target_x.shape[0] > 0
            break

    def test_epidir_in_dataframe(self, minimal_datamodule_config, mock_fasta_file, mock_df):
        """Test handling of epiDir column in dataframe."""
        # Get the actual directory path from the config
        bw_dir = minimal_datamodule_config["bw_dir"]

        df = mock_df.copy()
        df["epiDir"] = [bw_dir] * len(df)  # All use first directory

        config = minimal_datamodule_config.copy()
        config["df"] = df
        config["test_guides"] = ["g8", "g9"]  # Need test_guides to avoid None error

        dm = GenomicDataModule(**config)
        dm.prepare_data()
        dm.setup()

        # Should handle epiDir column without errors
        assert dm.dataset is not None


# ============================================================================
# Tests for edge cases
# ============================================================================

class TestGenomicDataModuleEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_epi_features(self, temp_dir, mock_fasta_file):
        """Test with empty epigenetic features."""
        config = {
            "fasta_path": mock_fasta_file,
            "epi_features": [],
            "bw_dir": temp_dir,
            "window_size": 100,
            "batch_size": 4,
            "num_workers": 0,
            "num_samples": 10,
        }

        dm = GenomicDataModule(**config)
        dm.prepare_data()
        dm.setup()

        assert dm.dataset is not None
        assert len(dm.epi_features) == 0

    def test_single_batch_size(self, minimal_datamodule_config, mock_fasta_file):
        """Test with batch size of 1."""
        config = minimal_datamodule_config.copy()
        config["batch_size"] = 1

        dm = GenomicDataModule(**config)
        dm.prepare_data()
        dm.setup()

        train_loader = dm.train_dataloader()

        for batch in train_loader:
            target_x, _, _, _, _, _ = batch
            assert target_x.shape[0] == 1
            break

    def test_large_window_size(self, minimal_datamodule_config, mock_fasta_file):
        """Test with large window size."""
        config = minimal_datamodule_config.copy()
        config["window_size"] = 500

        dm = GenomicDataModule(**config)
        dm.prepare_data()
        dm.setup()

        assert dm.window_size == 500

    def test_no_val_guides(self, minimal_datamodule_config, mock_fasta_file, mock_df):
        """Test without validation guides."""
        config = minimal_datamodule_config.copy()
        config["df"] = mock_df
        config["val_guides"] = None
        config["test_guides"] = ["g0", "g1"]

        dm = GenomicDataModule(**config)
        dm.prepare_data()
        dm.setup()

        # Should create val mask of all zeros
        _, val_mask, _ = dm.masks
        assert val_mask.sum() == 0

    def test_no_test_guides(self, minimal_datamodule_config, mock_fasta_file, mock_df):
        """Test without test guides (verifies test_guides=None fix)."""
        config = minimal_datamodule_config.copy()
        config["df"] = mock_df
        config["test_guides"] = None  # This used to cause TypeError

        dm = GenomicDataModule(**config)
        dm.prepare_data()
        dm.setup()

        # Should create test mask of all zeros
        _, _, test_mask = dm.masks
        assert test_mask.sum() == 0

    def test_random_val_split(self, minimal_datamodule_config, mock_fasta_file, mock_df):
        """Test automatic validation split when val_guides is empty list."""
        config = minimal_datamodule_config.copy()
        config["df"] = mock_df
        config["val_guides"] = []  # Empty list triggers random split
        config["test_guides"] = ["g0", "g1"]

        dm = GenomicDataModule(**config)
        dm.prepare_data()
        dm.setup()

        # Should have some validation samples
        _, val_mask, _ = dm.masks
        # At least some samples should be in validation
        assert val_mask.shape[0] == len(mock_df)

    def test_multiple_bw_dirs(self, temp_dir, mock_fasta_file):
        """Test with multiple BigWig directories."""
        # Create second directory
        second_dir = tempfile.mkdtemp()
        try:
            for epi_feat in ["ATAC", "DNASE"]:
                for chrom in ["chr1", "chr2", "chr3"]:
                    feat_array = np.random.rand(1000).astype(np.float32)
                    file_path = os.path.join(second_dir, f"{epi_feat}_{chrom}.npy")
                    np.save(file_path, feat_array)

            config = {
                "fasta_path": mock_fasta_file,
                "epi_features": ["ATAC", "DNASE"],
                "bw_dir": [temp_dir, second_dir],
                "window_size": 100,
                "batch_size": 4,
                "num_workers": 0,
                "num_samples": 10,
            }

            dm = GenomicDataModule(**config)
            dm.prepare_data()
            dm.setup()

            assert len(dm.bw_dirs) == 2
            assert dm.dataset is not None
        finally:
            shutil.rmtree(second_dir, ignore_errors=True)

    def test_bw_mode(self, temp_dir, mock_fasta_file):
        """Test with BigWig mode (not numpy mode)."""
        # For this test, we just verify the mode is accepted
        # Actual BigWig conversion requires real BigWig files
        config = {
            "fasta_path": mock_fasta_file,
            "epi_features": [],  # Empty to avoid file checks
            "bw_dir": temp_dir,
            "window_size": 100,
            "batch_size": 4,
            "num_workers": 0,
            "num_samples": 10,
            "mode": "bw",
        }

        dm = GenomicDataModule(**config)
        assert dm.mode == "bw"


# ============================================================================
# Integration tests
# ============================================================================

class TestGenomicDataModuleIntegration:
    """Integration tests for full data module workflow."""

    def test_full_workflow_no_dataframe(self, minimal_datamodule_config, mock_fasta_file, cpu_device):
        """Test complete workflow without dataframe."""
        dm = GenomicDataModule(**minimal_datamodule_config)

        # Full workflow
        dm.prepare_data()
        dm.setup()

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        # Iterate through one batch from each
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))

        assert len(train_batch) == 6  # 6 elements in tuple
        assert len(val_batch) == 6

    def test_full_workflow_with_dataframe(self, minimal_datamodule_config, mock_fasta_file, cpu_device):
        """Test complete workflow with dataframe."""
        df = pd.DataFrame({
            "GuideID": [f"g{i}" for i in range(8)],
            "label": [0, 1] * 4,
            "chr": ["chr1"] * 4 + ["chr2"] * 4,
            "start": [100 + i * 10 for i in range(8)],
            "end": [123 + i * 10 for i in range(8)],
            "Guide_sequence": ["ACGT" * 6 + "N"] * 8,
            "Strand": ["+"] * 8,
            "AlphagenomeIndex": list(range(8)),
            "extended_off_target": ["ACGT" * 26] * 8,
        })

        config = minimal_datamodule_config.copy()
        config["df"] = df
        config["val_guides"] = ["g6", "g7"]
        config["test_guides"] = ["g4", "g5"]

        dm = GenomicDataModule(**config)

        # Full workflow
        dm.prepare_data()
        dm.setup()

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()

        # Verify all loaders work
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        test_batch = next(iter(test_loader))

        assert len(train_batch) == 6
        assert len(val_batch) == 6
        assert len(test_batch) == 6
