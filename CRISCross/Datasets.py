import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import numpy as np
import collections
import torch.multiprocessing as mp
from typing import List, Dict, Tuple
import os
import shutil
import random
import pyBigWig
import warnings
warnings.filterwarnings("ignore", category=Warning, module="Bio")
from Bio import SeqIO
import math
import multiprocessing
from CRISCross.energyCalculations import get_eng, calcRNADNAenergy
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

mp.set_start_method("spawn", force=True)

def mutation_weights(window_size=23, max_mut=6, device="cpu"):
    k = torch.arange(1, max_mut + 1, device=device, dtype=torch.float64)

    weights = torch.tensor(
        [math.comb(window_size, int(i)) * (3 ** int(i)) for i in k],
        dtype=torch.float64,
        device=device
    )

    return weights / weights.sum()

MUT_WEIGHTS = mutation_weights(window_size=23, max_mut=6)

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

#TARGET_VOCAB = load_vocab("data/vocab.txt")

MAPPING = np.zeros(256, dtype=np.uint8)
MAPPING[ord("N")] = 0
MAPPING[ord("A")] = 1
MAPPING[ord("C")] = 2
MAPPING[ord("G")] = 3
MAPPING[ord("T")] = 4
COMP_MAPPING = np.array([0, 4, 3, 2, 1], dtype=np.uint8)
TORCHCOMPLEMENT = torch.tensor([0, 4, 3, 2, 1], dtype=torch.long)
REV_MAPPING = np.array([b'N', b'A', b'C', b'G', b'T', ], dtype='S1')
REVMAP_RNA =  np.array([b'N', b'A', b'C', b'G', b'U', ], dtype='S1')



def get_mapping(sequences):

    tensors = []
    # Convert string to NumPy array of ASCII codes
    for seq in sequences:
        ascii_vals = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
        genome_int = MAPPING[ascii_vals]

    # Convert to torch tensor
        tensor = torch.from_numpy(genome_int)
        tensors.append(tensor)
    return np.stack(tensors)

class SimpleDataset(Dataset):
    def __init__(self, indices, targets, off_targets, ag_indices, y, ag_feat, ag_dir,):
        self.indices = indices
        self.targets = targets
        self.off_targets = off_targets
        self.ag_indices = ag_indices
        self.y = y
        self.ag_feat = ag_feat
        self.ag_dir = ag_dir
        self.ag_memaps = {}
        for feature, shape in self.ag_feat.items():
            file = os.path.join(self.ag_dir, f"{feature}.np")
            mmap_array = np.memmap(file, dtype=np.float32, mode="r", shape=shape)
            self.ag_memaps[feature] = mmap_array


    def __len__(self):
        return 10000
    
    def __getitem__(self, i):
        idx = self.indices[i]
        target_x = self.targets[idx]
        return target_x
    

def shared_random_swaps(
    target_x,
    off_target_x,
    max_swaps=5,
):
    L = target_x.shape[0]

    target_x = target_x.clone()
    off_target_x = off_target_x.clone()

    n_swaps = torch.randint(1, max_swaps + 1, (1,)).item()

    for _ in range(n_swaps):
        i, j = torch.randperm(L)[:2]
        tmp = target_x[i].clone()
        target_x[i] = target_x[j]
        target_x[j] = tmp

        tmp = off_target_x[i].clone()
        off_target_x[i] = off_target_x[j]
        off_target_x[j] = tmp

    return target_x, off_target_x

class VeryHardDataset(Dataset):
    def __init__(self, indices, targets, off_targets, labels, ag_indices, strands, counts, ag_dir, ag_features, windowsize, augment: bool =False, epi_stats=None):
        self.targets = targets
        self.labels = labels
        self.counts = counts
        self.indices = indices
        self.augment = augment
        self.strands = strands
        self.off_targets = off_targets
        self.ag_indices = ag_indices
        self.ag_features = ag_features
        self.ag_indices = ag_indices
        self.ag_dir = ag_dir
        self.ag_memaps = {}
        self.windowsize = windowsize
        self.epi_stats = epi_stats
        if self.epi_stats:
            self.epi_mean = torch.tensor([s["mean"] for s in self.epi_stats])
            self.epi_std  = torch.tensor([s["std"] for s in self.epi_stats]).clamp_min(1e-6)
            self.epi_log_mask = torch.tensor([
            s["mode"] in ("COUNT", "SPARSE_HEAVY") for s in self.epi_stats])
        


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        target_x = self.targets[idx]
        off_target_x = self.off_targets[idx]
        strand = self.strands[idx]
        y = self.labels[idx]
        counts = self.counts[idx]
        nonzero_indices = torch.nonzero(counts)
        counts = counts[torch.randint(0, nonzero_indices.numel(), (1,))].squeeze() if nonzero_indices.numel() > 0 else counts[0]
        epi = []
        ag_index = self.ag_indices[idx]
        for feature, shape in self.ag_features.items():
            file = os.path.join(self.ag_dir, f"{feature}.np")
            # OPEN the memmap temporarily for this sample ONLY
            center = _default_size[0] // 2


            feat = np.memmap(file, dtype=np.float32, mode="r", shape=shape)
            if shape[1] < _default_size[0]:
                feat = feat[ag_index]
                factor = _default_size[0] // shape[1]
                feat = np.repeat(feat, factor, axis=0)

                feat = feat[center-self.windowsize // 2 - (self.windowsize % 2): center + self.windowsize // 2]

            else:

                feat = feat[ag_index, center-self.windowsize // 2 - (self.windowsize % 2): center + self.windowsize // 2]
            t = torch.from_numpy(feat.copy())
            epi.append(t)
        if len(epi):
            epi = torch.cat(epi, axis=-1)
            epi[epi.isnan()] = 0
        #epi = torch.log(epi + 1e-7)
            if self.epi_stats:
                epi[:, self.epi_log_mask] = torch.log1p(epi[:, self.epi_log_mask])
                epi = (epi - self.epi_mean) / self.epi_std
        else:
            epi = 0
        if self.augment:
            center = off_target_x.shape[0] // 2
            otc = off_target_x[center - 23//2 - 1:center+23//2]
            target_x, otc = shared_random_swaps(target_x=target_x, off_target_x=otc)
            off_target_x[center - 23//2 - 1:center+23//2] = otc
            

        return target_x, off_target_x, epi, y, counts, strand

_default_size = (2**14, 1)
histone_size = (128, 1)

EPI_FEATURES = {
            "ATAC": _default_size,
            "DNASE": _default_size, 
            "RNA_SEQ": (2 ** 14, 4),
            "EX_ATAC": _default_size,
            "EX_H3K4me1": _default_size,
            "EX_H3K4me3": _default_size,
            "EX_H3K9ac": _default_size,
            "EX_H3K9me3": _default_size,
            "EX_H3K27ac": _default_size,
            "EX_H3K27me3": _default_size,
            "EX_H3K36me3": _default_size,
            '+_polyA plus RNA-seq': _default_size,
            '+_total RNA-seq': _default_size,
            '-_polyA plus RNA-seq': _default_size,
            '-_total RNA-seq': _default_size,
            'H3K27ac': histone_size,
            'H3K27me3': histone_size,
            'H3K36me3': histone_size,
            'H3K4me1': histone_size,
            'H3K4me3': histone_size,
            'H3K9me3': histone_size,


        }
EPI_WEIGHTS = {
            "ATAC": (0.1,),
            "DNASE":(0.1,), 
            "RNA_SEQ": (0, 0, 0.1, 0.1),
            "EX_ATAC": (0.1,),
            "EX_H3K4me1": (0.1,),
            "EX_H3K4me3": (0.1,),
            "EX_H3K9ac": (0.1,),
            "EX_H3K9me3": (0.1,),
            "EX_H3K27ac": (0.1,),
            "EX_H3K27me3": (0.1,),
            "EX_H3K36me3": (0.1,),
            "CHIP_HISTONE": (0.01, 0.01, 0.01, 0.01, 0.01, 0.01),
            

        }


NORMALIZATION = {
    "DENSE": "zscore",
    "COUNT": "log1p+zscore",
    "SPARSE_HEAVY": "log1p+zscore",
}
eps = 1e-8


def estimate_energy_stats(loader):
    n = 0
    mean = 0.0
    M2 = 0.0  # sum of squared differences from the mean

    for batch in loader:
        energy = batch[4]  # index-safe: works for 6- and 7-element batches

        energy = energy.view(-1).float()  # flatten batch
        batch_n = energy.numel()

        batch_mean = energy.mean().item()
        batch_var = energy.var(unbiased=False).item()

        # merge batch statistics with running statistics
        delta = batch_mean - mean
        total_n = n + batch_n

        mean += delta * batch_n / total_n
        M2 += batch_var * batch_n + delta**2 * n * batch_n / total_n
        n = total_n

    std = (M2 / n) ** 0.5
    return mean, std
         # epi: (B, L, F)

def estimate_all_stats(loader):
    all_epi = []
    for idx, batch in enumerate(loader):
        epi = batch[2]          # index-safe: works for 6- and 7-element batches
        epi = epi.detach().cpu()
        all_epi.append(epi)

    # (N, F) where N = total_batch * seqlen
    all_epi = torch.cat(all_epi, dim=0).view(-1, all_epi[0].shape[-1])

    mean = all_epi.mean(dim=0)
    min_ = all_epi.min(dim=0).values
    max_ = all_epi.max(dim=0).values
    median = all_epi.median(dim=0).values
    qs = [0.01, 0.05, 0.25, 0.75, 0.95, 0.99]
    # quantiles (example)
    quantiles = torch.quantile(
        all_epi[:100000, :],
        q=torch.tensor(qs),
        dim=0
    )
    stats = []
    for i in range(all_epi.shape[1]):
        q25 = quantiles[2, i]
        q75 = quantiles[3, i]
        q99 = quantiles[5, i]
        max_v = max_[i]
        x = all_epi[:, i]
        tail_ratio = q99 / (q75 + eps)

        if max_v == 0:
            mode = "DROP"

        elif q25 > 0 and tail_ratio < 10:
            mode = "DENSE"

        elif q25 == 0 and tail_ratio > 10:
            mode = "SPARSE_HEAVY"

        else:
            mode = "COUNT"
        stats.append({
            "mean": mean[i],
            "std": x.std(dim=0, unbiased=False).clamp(min=1e-6),
            "median": median[i],
            "q25": q25,
            "q75": q75,
            "mode": mode
        })
    return stats



def estimate_stats(loader):
    feature_sum = None
    feature_sq_sum = None
    count = 0
    for batch in loader:
        epi = batch[2]  # index-safe: works for 5-, 6-, and 7-element batches

        # move to CPU if needed
        epi = epi.detach().cpu()

        B, L, F = epi.shape
        epi = epi.view(-1, F)  # (B*L, F)

        if feature_sum is None:
            feature_sum = epi.sum(dim=0)
            feature_sq_sum = (epi ** 2).sum(dim=0)
        else:
            feature_sum += epi.sum(dim=0)
            feature_sq_sum += (epi ** 2).sum(dim=0)

        count += epi.shape[0]

    mean = feature_sum / count
    var = (feature_sq_sum / count) - mean**2
    std = torch.sqrt(var)
    std[std == 0] = 1
    return mean, var, std

class MyDataModule(pl.LightningDataModule):
    def __init__(self, df,  val_guides, test_guides, exclude, ag_dir, windowsize, embedding_type, num_samples, epi_features, batch_size=32, oversample=True, norm_epi = False):
        super().__init__()
        self.df = df[~df["GuideID"].isin(exclude)].copy()
        self.oversample = oversample
        if "Score_norm" in self.df:
            self.df["Score_norm"] = self.df["Score_norm"].apply(eval)

        tmp_base = os.environ.get("TMPDIR", "/tmp")  # SLURM sets TMPDIR, fallback to /tmp
        self.local_ag_dir = os.path.join(tmp_base, os.path.basename(ag_dir))
        self.prepare_data_per_node = True

        self.batch_size = batch_size
        self.val_guides = val_guides
        self.test_guides = test_guides
        self.embedding_type = embedding_type
        self.tmp_dir = ag_dir
        self.ag_dir = self.local_ag_dir
        self.windowsize = windowsize
        self.num_samples = num_samples
        self.ag_features = {
            key: (int(self.df["AlphagenomeIndex"].max()) + 1, *EPI_FEATURES[key]) for key in epi_features

        }
        self.masks = None # contains train val test mask after setup
        self.norm_epi = norm_epi
        

    def _norm_epi(self, train_indices, all_targets, all_off_targets, y, ag_indices, counts, strands):
        dataset = VeryHardDataset(train_indices, all_targets, all_off_targets, y, ag_indices, strands, counts, self.ag_dir, self.ag_features, self.windowsize, augment=False)
        dl = DataLoader(dataset, sampler= None, batch_size=self.batch_size, num_workers=10, persistent_workers=False)
        stats = estimate_all_stats(dl)
        return stats

    def prepare_data(self):
        if not os.path.exists(self.local_ag_dir):
            print(f"Copying dataset  to {self.local_ag_dir}...")
            shutil.copytree(self.tmp_dir, self.local_ag_dir)
        else:
            print(f"Local dataset already exists at {self.local_ag_dir}, using it.")

    def setup(self, stage=None):
        # called on every GPU in distributed settings
        all_targets, all_off_targets, y, ag_indices, counts, strands = self.preprocess_data()
        sets = self.train_val_test_split(all_targets, all_off_targets, y, ag_indices)
        train_indices = torch.where(self.masks[0])[0]        
        val_indices = torch.where(self.masks[1])[0]        
        test_indices = torch.where(self.masks[2])[0]      
        train_indices.share_memory_()
        val_indices.share_memory_()
        test_indices.share_memory_()  
        all_targets.share_memory_()
        all_off_targets.share_memory_()
        y.share_memory_()
        ag_indices.share_memory_()
        if self.norm_epi:
            epi_stats = self._norm_epi(train_indices=train_indices, all_targets=all_targets,all_off_targets=all_off_targets, y = y, ag_indices=ag_indices, strands=strands, counts=counts)
        else:
            epi_stats = None
        self.train_dataset = VeryHardDataset(train_indices, all_targets, all_off_targets, y, ag_indices, strands, counts, self.ag_dir, self.ag_features, self.windowsize, augment=False, epi_stats=epi_stats, )
        self.val_dataset = VeryHardDataset(val_indices, all_targets, all_off_targets, y, ag_indices, strands, counts, self.ag_dir, self.ag_features, self.windowsize, epi_stats=epi_stats, )
        self.test_dataset = VeryHardDataset(test_indices, all_targets, all_off_targets, y, ag_indices, strands, counts, self.ag_dir, self.ag_features, self.windowsize, epi_stats=epi_stats, )
        #self.train_dataset = Ver(train_indices, all_targets, all_off_targets, ag_indices, y, self.ag_features, self.ag_dir)


    def compute_classweights(self, train_mask):
        labels = self.df[train_mask.numpy()]["label"]
        class_counts = labels.value_counts()
        weights = len(labels) / (len(class_counts) * class_counts)  # per class
        self.df.loc[:, "classweight"] = labels.map(weights)



    def train_val_test_split(self, all_targets, all_off_targets, y, ag_indices):
        test_mask = torch.from_numpy(self.df["GuideID"].isin(self.test_guides).values)
        if self.val_guides is None:
            val_mask = torch.zeros(test_mask.shape, dtype=test_mask.dtype)
        elif len(self.val_guides):
            val_mask = torch.from_numpy(self.df["GuideID"].isin(self.val_guides).values)
        else:
            val_mask = (torch.rand(test_mask.shape) < 0.2) & (~test_mask)

        train_mask = (~((val_mask) | (test_mask)))
        self.compute_classweights(train_mask)
        self.masks = [train_mask, val_mask, test_mask]

    def preprocess_data(self):
        df = self.df
        embedding_type = self.embedding_type
        
        
        # Process off-targets
        all_off_targets_np = get_mapping(df["extended_off_target"]) # <-- fresh NumPy array
        center = all_off_targets_np.shape[1] // 2
        all_off_targets_np = all_off_targets_np[:, center-self.windowsize//2 - (self.windowsize % 2): center+self.windowsize//2]
        all_off_targets = torch.tensor(all_off_targets_np, dtype=torch.uint8).contiguous()

        # Encode target sequences and create fresh tensor
        if embedding_type == "raw":
            all_targets_np = get_mapping(df["Guide_sequence"]) # <-- fresh NumPy array
            all_targets = torch.tensor(all_targets_np, dtype=torch.uint8).contiguous()
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")


        # Labels
        labels_np = df["label"].to_numpy(copy=True)    # <-- separate NumPy array
        y = torch.tensor(labels_np, dtype=torch.uint8).contiguous()
        if "Score_norm" in self.df:
            read_counts_np = df["Score_norm"].apply(lambda x: self.pad_and_convert(x, 3)).tolist()
            counts = torch.stack(read_counts_np).contiguous()

        else:
            counts = torch.zeros((len(self.df), 3), dtype=torch.uint8)
    # <-- separate NumPy array
        # Alphagenome indices
        ag_indices_np = df["AlphagenomeIndex"].to_numpy(copy=True)
        ag_indices = torch.tensor(ag_indices_np, dtype=torch.long).contiguous()
        strands = torch.tensor((df['Strand'] == '+').astype(int).values, dtype=torch.uint8)
        return all_targets, all_off_targets, y, ag_indices, counts, strands

    @staticmethod
    def pad_and_convert(scores, target_size):
        """Pads a list of scores with zeros and converts it to a PyTorch tensor."""
        # 1. Create a zeros tensor of the target size (e.g., [0., 0., 0.])
        padded_scores = torch.zeros(target_size, dtype=torch.float)

        # 2. Convert the input list to a tensor
        score_tensor = torch.tensor(scores, dtype=torch.float)

        # 3. Assign the actual scores to the beginning of the zeros tensor
        # The slice [:len(scores)] handles the variable length.
        padded_scores[:len(scores)] = score_tensor

        return padded_scores

    def train_dataloader(self):
        weights_np = self.df["classweight"].to_numpy(copy=True)[self.masks[0].numpy()]

        # Convert to torch tensor, make contiguous and share memory
        weights = torch.tensor(weights_np).contiguous().share_memory_()
        if self.oversample:
            sampler = WeightedRandomSampler(
                num_samples=self.num_samples,
                replacement=True,
                weights=weights
            )
        else:
            sampler=None
        return DataLoader(self.train_dataset, sampler= sampler, batch_size=self.batch_size, num_workers=5, persistent_workers=True, shuffle=True if sampler is None else False, drop_last=True if sampler is None else False, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=3, persistent_workers=True)

    def test_dataloader(self):
        # Optional: if you have a test set
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


def choose_num_mutations_torch():
    # returns 1..6
    return torch.multinomial(MUT_WEIGHTS, 1).item() + 1

def mutate_target(target, alphabet=(1, 2, 3, 4)):
    """
    target: torch.Tensor [window_size], dtype=torch.long or torch.uint8
    """
    device = target.device
    window_size = target.shape[0]

    num_mut = choose_num_mutations_torch(
    )

    # mutation positions
    mut_pos = torch.randperm(window_size, device=device)[:num_mut]

    current = target[mut_pos]

    alphabet = torch.tensor(alphabet, device=device, dtype=torch.long)
    new = alphabet[torch.randint(0, len(alphabet), (num_mut,), device=device)]

    # ensure new base != old base
    mask = new == current
    while mask.any():
        new[mask] = alphabet[
            torch.randint(0, len(alphabet), (mask.sum(),), device=device)
        ]
        mask = new == current

    target[mut_pos] = new
    return target


class GenomicDataset(Dataset):
    def __init__(self, chrom_sizes, seq_dict, bw_dir, epi_features, window_size=1000, num_samples=10000, epi_stats=None, atac_features=None, atac_stats=None):
        """
        chrom_sizes: dict of {chrom: length}, e.g., {'chr1': 248956422, ...}
        bigwig_paths: list of paths to bigwig files (each a feature track)
        window_size: length of the region to sample
        num_samples: number of random regions to generate
        atac_features: track names loaded as regression targets only (not model inputs)
        atac_stats: dict with "mean" and "std" tensors for ATAC normalization
        """
        self.chrom_sizes = chrom_sizes
        self.bw_dir = bw_dir
        self.epi_features = epi_features
        self.window_size = window_size
        self.num_samples = num_samples
        self.chroms = list(chrom_sizes.keys())
        self.seq_dict = seq_dict
        self.epi_stats = epi_stats
        if self.epi_stats:
            self.epi_mean = torch.tensor([s["mean"] for s in self.epi_stats])
            self.epi_std  = torch.tensor([s["std"] for s in self.epi_stats]).clamp_min(1e-6)
            self.epi_log_mask = torch.tensor([
                s["mode"] in ("COUNT", "SPARSE_HEAVY") for s in self.epi_stats])
        self.atac_features = atac_features or []
        self.atac_stats = atac_stats
        if self.atac_stats:
            self.atac_mean = torch.as_tensor(atac_stats["mean"], dtype=torch.float32)
            self.atac_std = torch.as_tensor(atac_stats["std"], dtype=torch.float32).clamp_min(1e-6)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Randomly choose a chromosome
        chrom = random.choices(self.chroms, weights=[self.chrom_sizes[c] for c in self.chroms], k=1)[0]
        max_start = self.chrom_sizes[chrom] - self.window_size
        if max_start <= 0:
            raise ValueError(f"Chromosome {chrom} is shorter than window_size")

        # Randomly choose a start position
        while True:
            start = random.randint(0, max_start)
            end = start + self.window_size
            off_target_x = self.seq_dict[chrom][start:end]
            center = off_target_x.shape[0] // 2 + off_target_x.shape[0] % 2
            if ~torch.all(off_target_x[center-23 // 2 - 1: center + 23 // 2] == 0):
                break
        off_target_x = off_target_x.long()
        if torch.rand(1) < 0.5:
            off_target_x = TORCHCOMPLEMENT[off_target_x.flip(-1)]
            strand = 0
        else:
            strand = 1
        mask = off_target_x == 0
        off_target_x[mask] = torch.randint(1, 5, (mask.sum(),), device=off_target_x.device, dtype=off_target_x.dtype)
        target_x = off_target_x[center-23 // 2 - 1: center + 23 // 2].clone()
        target_x = mutate_target(target_x)

        # Pick ONE bw_dir per sample so all tracks are from the same cell type
        bw_dir = np.random.choice(self.bw_dir)

        # Read histone/epi features for this region (used as model inputs)
        features = []
        for epi_feat in self.epi_features:
            file = os.path.join(bw_dir, f"{epi_feat}_{chrom}.npy")
            feat = np.memmap(file, dtype=np.float32, mode="r", shape=self.chrom_sizes[chrom])
            feat = feat[start:end]
            t = torch.from_numpy(feat.copy())
            features.append(t)
        if len(features):
            epi = torch.stack(features, axis=-1)  # [window_size, num_epi]
            epi.nan_to_num_(0)
            if self.epi_stats:
                epi[:, self.epi_log_mask] = torch.log1p(epi[:, self.epi_log_mask])
                epi = (epi - self.epi_mean) / self.epi_std
            if not strand:
                epi = epi.flip(0)  # flip spatial dimension for reverse complement
        else:
            epi = 0

        # Read ATAC features for this region (used as regression targets, not model inputs)
        atac_list = []
        for atac_feat in self.atac_features:
            file = os.path.join(bw_dir, f"{atac_feat}_{chrom}.npy")
            feat = np.memmap(file, dtype=np.float32, mode="r", shape=self.chrom_sizes[chrom])
            feat = feat[start:end]
            t = torch.from_numpy(feat.copy())
            atac_list.append(t)
        if atac_list:
            atac = torch.stack(atac_list, dim=-1)  # [window_size, num_atac]
            atac.nan_to_num_(0)
            atac = torch.log1p(atac)  # ATAC is always count/sparse — log1p stabilises scale
            if self.atac_stats:
                atac = (atac - self.atac_mean) / self.atac_std
            if not strand:
                atac = atac.flip(0)  # flip spatial dimension for reverse complement
        else:
            atac = torch.zeros(self.window_size, 0)

        y = 0
        counts = 0
        return target_x, off_target_x, epi, y, counts, strand, atac

    def close(self):
        for bw in self.bigwigs:
            bw.close()




class FineTuningGenomicDataset(Dataset):
    def __init__(self, centers, targets, labels, window_size, indices, strands, chrom_indices, chrom_sizes, chrom_mapping, seq_dict, bw_dir, epi_features, epi_stats = None, bw_indices = None):
        """
        bigwig_paths: list of paths to bigwig files (each a feature track)
        """
        self.chrom_indices = chrom_indices
        self.chrom_sizes = chrom_sizes
        self.bw_indices = bw_indices
        self.indices = indices
        self.centers = centers
        self.epi_features = epi_features
        self.strands = strands
        self.seq_dict = seq_dict
        self.epi_stats = epi_stats
        self.bw_dir = bw_dir
        self.window_size = window_size
        self.chrom_mapping =chrom_mapping
        self.labels = labels
        self.targets = targets
        if self.epi_stats:
            self.epi_mean = torch.tensor([s["mean"] for s in self.epi_stats])
            self.epi_std  = torch.tensor([s["std"] for s in self.epi_stats]).clamp_min(1e-6)
            self.epi_log_mask = torch.tensor([
                s["mode"] in ("COUNT", "SPARSE_HEAVY") for s in self.epi_stats])

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        idx = self.indices[i]
        
        y = self.labels[idx]

        chrom_idx = self.chrom_indices[idx]
        chrom = self.chrom_mapping[int(chrom_idx.item())]
        strand = self.strands[idx]
        center = self.centers[idx]

 
        start = center - self.window_size // 2 - (self.window_size % 2)
        end = center + self.window_size // 2
        off_target_x = self.seq_dict[chrom][start:end]
        target_x = self.targets[idx]
        off_target_x = off_target_x.long()
        if not strand:
            off_target_x = TORCHCOMPLEMENT[off_target_x.flip(-1)]
                # Read each BigWig feature for this region
        features = []
        if self.bw_indices is None:
            bw_dir = self.bw_dir[0]
        else:
            bw_dir = self.bw_dir[self.bw_indices[idx]]
            
        for epi_feat in self.epi_features:
            file = os.path.join(bw_dir, f"{epi_feat}_{chrom}.npy")
            feat = np.memmap(file, dtype=np.float32, mode="r", shape=self.chrom_sizes[chrom])
            feat = feat[start:end]
            t = torch.from_numpy(feat.copy())
            features.append(t)
        if len(features):
            epi = torch.stack(features, axis=-1)
            epi.nan_to_num_(0)
            if self.epi_stats:
                epi[:, self.epi_log_mask] = torch.log1p(epi[:, self.epi_log_mask])
                epi = (epi - self.epi_mean) / self.epi_std
            if not strand:
                epi = epi.flip(0)  # flip spatial dimension for reverse complement
        else:
            epi = 0
        counts = 0
        return target_x, off_target_x, epi, y, counts, strand



class EnergyGenomicDataset(GenomicDataset):
    def __init__(self, *args, **kwargs): 
        if "energy_stats" in kwargs:
            self.energy_stats = kwargs["energy_stats"]
            del kwargs["energy_stats"]

        else:
            self.energy_stats = None
        super().__init__( *args, **kwargs)
    
    def __getitem__(self, idx):
        target_x, off_target_x, epi, y, counts, strands, atac = super().__getitem__(idx)
        tx = target_x.cpu().numpy()
        target_seq = b''.join(REV_MAPPING[tx]).decode('ascii')       # reverse            # complement                 # move to CPU if needed
        #target_seq = REVMAP_RNA[COMP_MAPPING[target_x.cpu()].flip(0).numpy()]
        #target_seq = b''.join(target_seq).decode('ascii')
 
        center = off_target_x.shape[0] // 2 + off_target_x.shape[0] % 2
    
        assert (off_target_x[center-23 // 2 - 1: center + 23 // 2] == target_x).sum() >= 15
        off_target_seq = REV_MAPPING[off_target_x[center-23 // 2 - 1: center + 23 // 2].cpu().numpy()]
        off_target_seq = b''.join(off_target_seq).decode('ascii')
        energy = get_eng(
            target_seq, off_target_seq,
            calcRNADNAenergy,
            GU_allowed=False,
            pos_weight=True,
            pam_corr=True,
            grna_folding=False,
            dna_opening=True,
            dna_pos_wgh=False
        )
        if self.energy_stats:
            energy = (energy - self.energy_stats[0]) / self.energy_stats[1]

        return target_x, off_target_x, epi, y, energy, strands, atac



FASTA_CACHE = "encoded_fasta.pt"
import pickle 



def convert_bigwig_to_memmap_per_chr(bw_path, out_dir):
    fname = os.path.splitext(os.path.basename(bw_path))[0]
    print(f"Processing {bw_path}")

    with pyBigWig.open(bw_path) as bw:
        chroms = bw.chroms()  # dict: {chrom_name: length}

        for chrom, length in chroms.items():
            memmap_path = os.path.join(out_dir, f"{fname}_{chrom}.npy")
            if os.path.exists(memmap_path):
                continue
            print(f"  Writing {memmap_path} ({length} bases)")

            # create memmap array
            arr = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=(length,))

            # load chromosome values in one call
            vals = bw.values(chrom, 0, length, numpy=True)
            vals = np.nan_to_num(vals, nan=0.0)  # replace NaN with 0
            arr[:] = vals

            arr.flush()
    print(f"Done: {bw_path}\n")

CHROMOSOME_SIZES = {
    "chr1": 248956422,
    "chr2": 242193529,
    "chr3": 198295559,
    "chr4": 190214555,
    "chr5": 181538259,
    "chr6": 170805979,
    "chr7": 159345973,
    "chr8": 145138636,
    "chr9": 138394717,
    "chr10": 133797422,
    "chr11": 135086622,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16": 90338345,
    "chr17": 83257441,
    "chr18": 80373285,
    "chr19": 58617616,
    "chr20": 64444167,
    "chr21": 46709983,
    "chr22": 50818468,
    "chrX": 156040895,
    "chrY": 57227415
}




class GenomicDataModule(pl.LightningDataModule):
    def __init__(self, fasta_path, epi_features, bw_dir, window_size=512, batch_size=32, num_workers=4, num_samples=10000, norm_epi=False, use_energy=False, mode="np", df=None, val_guides=None, test_guides=None, atac_features=None, norm_num_samples=10000):
        super().__init__()
        self.chrom_sizes = None
        self.seq_dict = None
        self.epi_features = epi_features
        self.use_energy = use_energy
        self.mode = mode
        self.fasta_path = fasta_path
        self.df = df
        if self.df is not None:
            self.oversample = True
            self.val_guides = val_guides
            self.test_guides = test_guides
        else:
            self.oversample = False
        if isinstance(bw_dir, str):
            bw_dir = [bw_dir]
        self.bw_dirs = bw_dir
        # atac_features must be set before bw_files so ATAC files are included in the copy list
        self.atac_features = atac_features or []

        self.chromosomes = list(CHROMOSOME_SIZES.keys())
        all_track_features = list(epi_features) + self.atac_features
        if self.mode == "bw":
            self.bw_files = [
                os.path.join(bw_dir, f"{feat}.bw") for feat in all_track_features for bw_dir in self.bw_dirs
            ]
        elif self.mode == "np":
            self.bw_files = [
                os.path.join(bw_dir, f"{feat}_{chr}.npy")
                for feat in all_track_features
                for chr in self.chromosomes
                for bw_dir in self.bw_dirs
                if os.path.exists(os.path.join(bw_dir, f"{feat}_{chr}.npy"))
            ]
            if len(epi_features):
                assert len(self.bw_files), "Path probably wrong"
        else:
            raise ValueError("mode must be either bw (bigwig) or np (numpy)")
        self.tmp_base = os.environ.get("TMPDIR", "/tmp")
        self.local_bw_dirs = [os.path.join(self.tmp_base, os.path.basename(bw_dir)) for bw_dir in self.bw_dirs]
        self.prepare_data_per_node = True

        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_samples = num_samples
        self.norm_epi = norm_epi
        self.norm_num_samples = norm_num_samples

    def _norm_eng(self):
        dataset = EnergyGenomicDataset(self.chrom_sizes, self.seq_dict, self.local_bw_dirs, self.epi_features, self.window_size, self.norm_num_samples)
        dl = DataLoader(dataset, sampler= None, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False)
        stats = estimate_energy_stats(dl)
        return stats

    
    def _norm_epi(self):
        dataset = GenomicDataset(self.chrom_sizes, self.seq_dict, self.local_bw_dirs, self.epi_features, self.window_size, self.norm_num_samples)
        dl = DataLoader(dataset, sampler=None, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False)
        stats = estimate_all_stats(dl)
        return stats

    def _norm_atac(self):
        # Estimate log1p mean/std for ATAC tracks using a small sample.
        # No epi input features needed here — we only care about the ATAC target values.
        dataset = GenomicDataset(
            self.chrom_sizes, self.seq_dict, self.local_bw_dirs,
            epi_features=[],
            window_size=self.window_size,
            num_samples=self.norm_num_samples,
            atac_features=self.atac_features,
            atac_stats=None,  # no normalisation yet — collect raw log1p values
        )
        dl = DataLoader(dataset, sampler=None, batch_size=self.batch_size, num_workers=min(4, self.num_workers), persistent_workers=False)
        all_atac = []
        for batch in dl:
            atac = batch[6].detach().cpu()  # [B, window_size, num_atac] — already log1p'd in dataset
            all_atac.append(atac.reshape(-1, atac.shape[-1]))
        all_atac = torch.cat(all_atac, dim=0)  # [N, num_atac]
        return {
            "mean": all_atac.mean(dim=0),
            "std": all_atac.std(dim=0).clamp_min(1e-6),
        }
    

    def prepare_data(self):
         # SLURM sets TMPDIR, fallback to /tmp
        local_bw_dirs = self.local_bw_dirs
        print(f"Generating dataset from {self.bw_dirs} to {self.local_bw_dirs}...")

        for local_bw_dir in local_bw_dirs:
            args = [(bw_path, local_bw_dir) for bw_path in self.bw_files]
            os.makedirs(local_bw_dir, exist_ok=True)
            if self.mode == "bw":
                with multiprocessing.Pool(self.num_workers) as pool:
                    pool.starmap(convert_bigwig_to_memmap_per_chr, args)
            elif self.mode == "np":
                for src in self.bw_files:
                    dst = os.path.join(local_bw_dir, os.path.basename(src))
                    if os.path.exists(dst):
                        continue
                    shutil.copy2(src, dst)
        print("Finished dataset generation")


        
    
        if os.path.exists(FASTA_CACHE):
            pass
        else:
            # Read FASTA and encode sequences
            seq_dict = {}
            chrom_sizes = {}
            for record in SeqIO.parse(self.fasta_path, "fasta"):
                chrom = record.id
                if chrom in self.chromosomes:
                    seq = str(record.seq).upper().encode("ascii")
                    encoded = torch.tensor(MAPPING[np.frombuffer(seq, dtype=np.uint8)], dtype=torch.uint8)
                    seq_dict[chrom] = encoded
                    chrom_sizes[chrom] = len(seq)
                    for bw_dir in self.bw_dirs:
                        for feat in self.epi_features:
                            file = os.path.join(bw_dir, f"{feat}_{chrom}.npy")
                            assert os.path.exists(file), f"{file} memmap track not found"
        
            # Save to disk for fast rerun
            with open(FASTA_CACHE, "wb") as f:
                pickle.dump({"seq_dict": seq_dict, "chrom_sizes": chrom_sizes}, f)
        return True

    def setup(self, stage=None):
    # If cache exists, load everything
        with open(FASTA_CACHE, "rb") as f:
            cache = pickle.load(f)
            self.seq_dict = cache["seq_dict"]
            self.chrom_sizes = cache["chrom_sizes"]


        for key, tensor in self.seq_dict.items():
            tensor.share_memory_()

        if self.norm_epi:
            stats = self._norm_epi()
        else:
            stats = None

        if self.use_energy:
            energy_stats = self._norm_eng()
            kwargs = {"energy_stats": energy_stats}
        else:
            kwargs = {}

        atac_stats = self._norm_atac() if self.atac_features else None

        if self.df is not None:
            print("Using Fine Tuning dataset via a dataframe")
            all_targets, centers, y, counts, strands, idx_to_chrom, chrom_indices, bw_indices = self.preprocess_data()
            self.train_val_test_split()
            train_indices = torch.where(self.masks[0])[0]        
            val_indices = torch.where(self.masks[1])[0]        
            test_indices = torch.where(self.masks[2])[0]
            self.dataset = FineTuningGenomicDataset(
                centers=centers,
                targets=all_targets,
                window_size=self.window_size,
                indices=train_indices,
                strands=strands,
                chrom_indices=chrom_indices,
                chrom_mapping=idx_to_chrom,
                bw_dir=self.local_bw_dirs,
                epi_features=self.epi_features,
                epi_stats=stats,
                seq_dict=self.seq_dict,
                chrom_sizes=self.chrom_sizes,
                labels=y,
                bw_indices=bw_indices

            )  
            self.val_set = FineTuningGenomicDataset(
                centers=centers,
                window_size=self.window_size,
                targets=all_targets,
                indices=val_indices,
                strands=strands,
                chrom_indices=chrom_indices,
                chrom_mapping=idx_to_chrom,
                bw_dir=self.local_bw_dirs,
                epi_features=self.epi_features,
                epi_stats=stats,
                seq_dict=self.seq_dict,
                chrom_sizes=self.chrom_sizes,
                labels=y,
                bw_indices=bw_indices



            )   
            self.test_set = FineTuningGenomicDataset(
                centers=centers,
                window_size=self.window_size,
                indices=test_indices,
                strands=strands,
                chrom_indices=chrom_indices,
                chrom_mapping=idx_to_chrom,
                bw_dir=self.local_bw_dirs,
                epi_features=self.epi_features,
                epi_stats=stats,
                targets=all_targets,
                seq_dict=self.seq_dict,
                chrom_sizes=self.chrom_sizes,
                labels=y,
                bw_indices=bw_indices




            )   
            
            

        else:

            df_class = EnergyGenomicDataset if self.use_energy else GenomicDataset

            self.dataset = df_class(
                chrom_sizes=self.chrom_sizes,
                seq_dict=self.seq_dict,
                bw_dir=self.local_bw_dirs,
                epi_features=self.epi_features,
                window_size=self.window_size,
                num_samples=self.num_samples,
                epi_stats=stats,
                atac_features=self.atac_features,
                atac_stats=atac_stats,
                **kwargs
            )
            self.val_set = df_class(
                chrom_sizes=self.chrom_sizes,
                seq_dict=self.seq_dict,
                bw_dir=self.local_bw_dirs,
                epi_features=self.epi_features,
                window_size=self.window_size,
                num_samples=1,
                epi_stats=stats,
                atac_features=self.atac_features,
                atac_stats=atac_stats,
                **kwargs
            )
            self.test_set = df_class(
                chrom_sizes=self.chrom_sizes,
                seq_dict=self.seq_dict,
                bw_dir=self.local_bw_dirs,
                epi_features=self.epi_features,
                window_size=self.window_size,
                num_samples=1,
                epi_stats=stats,
                atac_features=self.atac_features,
                atac_stats=atac_stats,
                **kwargs
            )

        train_len = len(self.dataset)
        val_len = len(self.val_set)
        print(
            f"[DATA] setup() complete -> train samples: {train_len}, val samples: {val_len}, "
            f"epi_features: {self.epi_features}, atac_features: {self.atac_features}"
        )
        if train_len == 0:
            raise RuntimeError(
                "GenomicDataModule.setup(): training dataset is empty (0 samples). "
                "Check num_samples, bw_dir paths, and fasta_path in the config."
            )

    def train_dataloader(self):
       
        if self.oversample:
            weights_np = self.df["classweight"].to_numpy(copy=True)[self.masks[0].numpy()]

            weights = torch.tensor(weights_np).contiguous().share_memory_()
            sampler = WeightedRandomSampler(
                num_samples=self.num_samples,
                replacement=True,
                weights=weights
            )
        else:
            sampler=None
        return DataLoader(
            self.dataset, 
            sampler= sampler, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True if sampler is None else False, 
            drop_last=True if sampler is None else False, 
            pin_memory=True
            )

    def val_dataloader(self):
        # Optionally you can create a separate validation dataset
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=max(self.num_workers // 2, 1),
            persistent_workers=True,
            pin_memory=True

            
        )
        
    def test_dataloader(self):
        # Optionally you can create a separate validation dataset
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
        )

        
    def preprocess_data(self):
        df = self.df
        dir_to_idx = {d: i for i, d in enumerate(self.bw_dirs)}
        if "epiDir" in self.df:
            bw_indices = self.df["epiDir"].map(dir_to_idx)
            bw_indices = torch.tensor(bw_indices.astype(int).values, dtype=torch.uint8)
            bw_indices.share_memory_()
            print("Using different epiTracks depending on sample")
        else:
            bw_indices = None
        
        strands = torch.tensor((df['Strand'] == '+').astype(int).values, dtype=torch.uint8)
        centers = (self.df["end"] + self.df["start"]) // 2

        centers = torch.tensor(centers, dtype=torch.long).contiguous()
        centers[strands.bool()] += 1

        chroms = sorted(self.df["chr"].unique())  # e.g., ['chr1', 'chr2', ..., 'chrX', 'chrY']
        chrom_to_idx = {c: i for i, c in enumerate(chroms)}
        idx_to_chrom = {i: c for c, i in chrom_to_idx.items()}
        


        # Map column to integer indices
        chrom_indices = torch.tensor(self.df["chr"].map(chrom_to_idx).to_numpy())

        
        
        all_targets_np = get_mapping(df["Guide_sequence"]) # <-- fresh NumPy array
        all_targets = torch.tensor(all_targets_np, dtype=torch.uint8).contiguous()
        # Labels
        labels_np = df["label"].to_numpy(copy=True)    # <-- separate NumPy array
        y = torch.tensor(labels_np, dtype=torch.uint8).contiguous()
        if "Score_norm" in self.df:
            read_counts_np = df["Score_norm"].apply(lambda x: self.pad_and_convert(x, 3)).tolist()
            counts = torch.stack(read_counts_np).contiguous()
        else:
            counts = torch.zeros((len(self.df), 3), dtype=torch.uint8)
        all_targets.share_memory_()
        strands.share_memory_()
        chrom_indices.share_memory_()
        centers.share_memory_()
        return all_targets, centers, y, counts, strands, idx_to_chrom, chrom_indices, bw_indices
    
    def train_val_test_split(self):
        if self.test_guides is None:
            test_mask = torch.zeros(len(self.df), dtype=torch.bool)
        else:
            test_mask = torch.from_numpy(self.df["GuideID"].isin(self.test_guides).values)
        if self.val_guides is None:
            val_mask = torch.zeros(test_mask.shape, dtype=test_mask.dtype)
        elif len(self.val_guides):
            val_mask = torch.from_numpy(self.df["GuideID"].isin(self.val_guides).values)
        else:
            val_mask = (torch.rand(test_mask.shape) < 0.2) & (~test_mask)

        train_mask = (~((val_mask) | (test_mask)))
        self.compute_classweights(train_mask)
        self.masks = [train_mask, val_mask, test_mask]
        
    def compute_classweights(self, train_mask):
        labels = self.df[train_mask.numpy()]["label"]
        class_counts = labels.value_counts()
        weights = len(labels) / (len(class_counts) * class_counts)  # per class
        self.df.loc[:, "classweight"] = labels.map(weights)
