import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import BinaryAveragePrecision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import torch.multiprocessing as mp
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
import multiprocessing.util
import collections
from typing import Dict, List, Tuple
from CRISCross.models import CRISCross
from CRISCross.Datasets import GenomicDataModule, EPI_FEATURES, MAPPING, EPI_WEIGHTS
from CRISCross.bulges import SITE_LEN, band_bounds, band_width, identity_alignment
import json
from torchmetrics import Metric
from torchmetrics.functional import spearman_corrcoef
from torchmetrics.classification import MulticlassAveragePrecision

from torch.optim.lr_scheduler import LambdaLR
import hashlib
from pytorch_lightning.strategies import DDPStrategy


mp.set_start_method("spawn", force=True)


def short_hash(epi_features, length=6):
    # deterministic string representation
    s = ",".join(sorted(epi_features))
    h = hashlib.md5(s.encode()).hexdigest()
    return h[:length]

def get_logger(config):
    epi_hash = short_hash(config["epi_features"])
    # Bulge settings go in the run directory name so that the points of the
    # bulge-rate sweep do not overwrite each other's checkpoints. Omitted
    # entirely when unset, so existing mismatch-only run paths are unchanged.
    bulge_tag = ""
    if config.get("band_delta") or config.get("bulge_rate") is not None or config.get("bulge_profile"):
        bulge_tag = f"_bd{config.get('band_delta', 0)}_br{config.get('bulge_rate', 'prof')}"
    base_dir = f"RUNlogs/{config['experiment']}/test_split{config['split']}/ctl{config['context_layers']}_bs{config['batch_size']}_ws{config['windowsize']}_ue{config['num_epi']}_seed{config['seed']}_energy{config['use_energy']}_hash{epi_hash}{bulge_tag}"
    run_dir = os.path.join(base_dir, "run_")
    existing = os.listdir(run_dir) if os.path.exists(run_dir) else []
    version = f"v{len(existing)}"
    print(f"[LOGGER] BASE DIR: {base_dir}")
    print(f"[LOGGER] Found {len(existing)} existing run(s) in {run_dir}, using version={version}")
    logger = TensorBoardLogger(
        save_dir=base_dir,   # your custom folder
        name="run_",
        version=version,
    )
    return logger

class PreTrainModel(pl.LightningModule):
    """Masked-token pretraining on CRISPRAT-style synthetic guide/target pairs.

    Bulge handling
    --------------
    Batches may carry three extra fields produced by
    :class:`CRISCross.Datasets.BulgeGenomicDataset`: the ground-truth
    ``align`` map, the ``edits`` script, and a ``gapless`` flag. All three are
    *labels*. When they are absent the module falls back to the gapless
    diagonal, which makes every code path below collapse onto the original
    mismatch-only computation exactly.

    Two invariants are worth spelling out, because breaking either would leak
    the bulge configuration into the model input:

    - The off-target band and the epigenetic track are masked along the **fixed
      diagonal**, never along the true alignment. Masking along the true
      alignment would make the guide and band zero-patterns line up, and the
      model could read the alignment straight off its own input.
    - The masking probability does not depend on whether a guide position is
      paired. Unpaired positions (RNA bulges) are masked at the same rate as any
      other and are excluded from the *loss* instead.
    """

    def __init__(self, context_layers, hidden_dim, num_epi, dropout, seed, windowsize, merge, epi_weights, lr=1e-4, borders=None, use_energy=False, num_atac=0, atac_weight=0.1, band_delta=0):
        super().__init__()

        if borders is not None:
            raise NotImplementedError("Not yet implemented")
            self.criterion = BarDistributionConfig(full_support=True, borders=borders).get_criterion()
            self.output_size = len(borders) - 1
            self.regression = True
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")
            self.output_size = 1
            self.regression = False
        self.use_energy = use_energy
        self.vocab_size = 5
        self.model = CRISCross(
            vocab_size=self.vocab_size,
            dropout=dropout,
            context_layers=context_layers,
            hidden_dim=hidden_dim,
            num_epi=num_epi,
            output_size=self.output_size,
            windowsize=windowsize,
            merge=merge,
            band_delta=band_delta,

        )
        self.windowsize = windowsize
        self.band_delta = band_delta
        self.band_start, self.band_end = band_bounds(windowsize, band_delta)
        self.band_w = band_width(band_delta)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.hparams.n_trainable_params = n_params
        self.hparams.seed = seed
        self.save_hyperparameters()


        self.lr = lr
        self.auprc = BinaryAveragePrecision()  # torchmetrics AUPRC
        self.test_auprc = BinaryAveragePrecision()
        self.alpha = nn.Parameter(torch.tensor(epi_weights), requires_grad=False)
        self.extra_epi_mask = False


        self.mask_prob = 0.2
        self.num_epi = num_epi

        self.max_idx = MAPPING.max()
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.epi_loss_fn = nn.MSELoss(reduction="none")
        self.per_nt_classifier = nn.Linear(hidden_dim, 25)
        self.per_nt_epi_head = nn.Linear(hidden_dim, num_epi)
        self.auprc = MulticlassAveragePrecision(num_classes=25)
        self.train_auprc = MulticlassAveragePrecision(num_classes=25)
        self._first_batches_features = []

        self.num_atac = num_atac
        self.atac_weight = atac_weight
        self.atac_head = nn.Linear(hidden_dim, num_atac) if num_atac > 0 else None
        self.atac_loss_fn = nn.MSELoss()
        # Per-sample so the hybrid-energy loss can be masked on bulged samples.
        self.energy_loss_fct = nn.MSELoss(reduction="none")



    
    def forward(self, target_x, off_target_x, epi, strands):
        cls_logits, hidden = self.model(target_x, off_target_x, strands, epi)
        epi_logits = self.per_nt_epi_head(hidden)
        # Mean-pool per-nt hidden states → [B, hidden_dim] → [B, num_atac] scalar prediction
        atac_logits = self.atac_head(hidden.mean(dim=1)) if self.atac_head is not None else None
        logits = self.per_nt_classifier(hidden)
        return logits, epi_logits, cls_logits, atac_logits
    
    def unpack_batch(self, batch):
        """Split a batch into the seven core fields and the bulge labels.

        Batches from :class:`~CRISCross.Datasets.GenomicDataset` have seven
        fields; the bulge-aware sampler appends ``align``, ``edits`` and
        ``gapless``. Missing labels are filled in with their gapless values, so
        callers never have to branch on the sampler.

        Args:
            batch: The tuple produced by the dataloader.

        Returns:
            ``(core, align, edits, gapless)`` where ``core`` is the original
            seven-tuple, ``align`` is ``[B, 23]`` band columns per guide
            position (``-1`` = unpaired), ``edits`` is ``[B, 3]`` or ``None``,
            and ``gapless`` is a float ``[B]`` mask.
        """
        core = tuple(batch[:7])
        target_x = core[0]
        device = target_x.device
        batch_size = target_x.shape[0]

        if len(batch) > 7:
            align = batch[7].long()
        else:
            align = identity_alignment(self.band_w, device).unsqueeze(0).expand(batch_size, -1)
        edits = batch[8].long() if len(batch) > 8 else None
        if len(batch) > 9:
            gapless = batch[9].to(device=device, dtype=torch.float).view(-1)
        else:
            gapless = torch.ones(batch_size, device=device)
        return core, align, edits, gapless

    def gather_band(self, band, align):
        """Pick, for every guide position, the band column it is paired with.

        Args:
            band: ``[B, band_w]`` (or ``[B, band_w, F]``) band tensor.
            align: ``[B, 23]`` band column indices; ``-1`` for unpaired.

        Returns:
            ``[B, 23]`` (or ``[B, 23, F]``). Unpaired entries take the value at
            column 0 and must be discarded by the caller via ``align >= 0``.
        """
        safe = align.clamp_min(0)
        if band.dim() == 3:
            safe = safe.unsqueeze(-1).expand(-1, -1, band.shape[-1])
        return torch.gather(band, 1, safe)

    def diagonal_columns(self, batch_size, device):
        """The fixed gapless diagonal, ``[B, 23]``.

        Used for everything that touches the model *input*. It carries no
        sample-specific information, so it cannot leak the bulge configuration.
        """
        return identity_alignment(self.band_w, device).unsqueeze(0).expand(batch_size, -1)

    def _scatter_mask(self, columns, values, width):
        """One-hot scatter of ``values`` ``[B, 23]`` onto ``[B, width]`` columns."""
        onehot = F.one_hot(columns, width).bool() & values.unsqueeze(-1)
        return onehot.any(dim=1)

    def mask_shit(self, batch):
        core, align, edits, gapless = self.unpack_batch(batch)
        target_x, off_target_x, epi = core[0], core[1], core[2]
        batch_size, seq_len = target_x.shape
        bs, be = self.band_start, self.band_end
        target_x = target_x.clone()

        # Generate a random mask. Note this is uniform over all guide positions,
        # including unpaired ones: the masking rate must not depend on the bulge
        # configuration.
        mask_tensor = (torch.rand(batch_size, seq_len, device=target_x.device) < self.mask_prob)
        empty = mask_tensor.sum(dim=1) == 0
        if empty.any():
            idx = torch.randint(seq_len, (empty.sum(),), device=target_x.device)
            mask_tensor[empty, idx] = True
        # Expand mask to hidden dimension
        rand = torch.rand(batch_size, seq_len, device=target_x.device)
        mask_mask = mask_tensor & (rand < 0.8)
        random_mask = mask_tensor & (rand >= 0.8) & (rand < 0.9)


        # Apply mask: set masked positions to zero
        target_x = target_x.masked_fill(mask_tensor, 0)
        random_tokens = torch.randint(
            low=1,
            high=self.vocab_size,
            size=target_x.shape,
            device=target_x.device,
            dtype=target_x.dtype
        )
        target_x[random_mask] = random_tokens[random_mask]

        # The band is masked along the FIXED diagonal, not along the true
        # alignment: a band mask that followed the alignment would let the model
        # recover the alignment by lining up the two zero patterns.
        diag = self.diagonal_columns(batch_size, target_x.device)
        off_target_x = off_target_x.clone()
        band_mask = self._scatter_mask(diag, mask_mask, self.band_w)
        off_target_x[:, bs:be] = off_target_x[:, bs:be].masked_fill(band_mask, 0)

        epi = epi.clone()
        if len(epi.shape) > 1:
            epi_band_mask = self._scatter_mask(diag, mask_tensor, self.band_w)
            epi[:, bs:be] = epi[:, bs:be].masked_fill(epi_band_mask.unsqueeze(-1), 0)
            d = torch.randint(low=0, high=min(32, self.windowsize // 2), size=(1,))
            B, T = epi.shape[0:2]
            if self.extra_epi_mask:
                center = self.band_end - SITE_LEN // 2
                d = torch.randint(min(self.windowsize // 2, 128 // 2), min(128 // 2, self.windowsize // 2)+1, (B,), device=epi.device)
                idx = torch.arange(T, device=epi.device).unsqueeze(0)
                epi_mask = (idx >= (center - d).unsqueeze(1)) & (idx < (center + d).unsqueeze(1))

                epi[epi_mask] = 0
        else:
            epi_mask = torch.zeros(epi.shape)
        return target_x, off_target_x, epi, mask_tensor


    def compute_tokenized_target(self, target_x, off_target_x, mask, align=None):
        """Build the paired (guide base, target base) MLM label.

        The target base is read at the position the guide position is *aligned
        to*, which is the diagonal for gapless samples and a shifted diagonal
        after a bulge. The label is only meaningful where the guide position has
        a partner, so ``mask`` should already exclude unpaired positions.

        Args:
            target_x: ``[B, 23]`` unmasked guide.
            off_target_x: ``[B, window]`` unmasked window.
            mask: ``[B, 23]`` boolean supervision mask.
            align: ``[B, 23]`` band columns; defaults to the gapless diagonal.

        Returns:
            ``[B, 23]`` long label tensor.
        """
        bs, slen = target_x.shape
        if align is None:
            align = self.diagonal_columns(bs, target_x.device)
        band = off_target_x[:, self.band_start:self.band_end]
        aligned_ot = self.gather_band(band, align)

        y1 = torch.zeros((bs, slen), dtype=torch.long).to(target_x.device)
        y2 = torch.zeros((bs, slen), dtype=torch.long).to(target_x.device)
        y1[mask] = target_x[mask].to(torch.long)
        y2[mask] = aligned_ot[mask].to(torch.long)

        #y1[y1 > 1] -= 2
        #y2[y2 > 1] -= 2
        y = y1 * self.max_idx + y2
        return y




    def general_step(self, batch):
        core, align, edits, gapless = self.unpack_batch(batch)
        target_x, off_target_x, epi, y, counts, strands, atac = core
        masked_target, masked_ot, epi_masked, mask = self.mask_shit(batch)

        band = off_target_x[:, self.band_start:self.band_end]
        aligned_ot = self.gather_band(band, align)
        paired = align >= 0
        # Identity along the ground-truth alignment. Bounded below by
        # 23 - (m + b_R) >= 17 for a sample within the edit budget.
        assert ((aligned_ot == target_x) & paired).sum(axis=1).min() >= 15

        # Unpaired guide positions (RNA bulges) have no target partner, so the
        # paired-token label is undefined there: mask the loss, not the input.
        mask = mask & paired

        bs, slen = target_x.shape
        y = self.compute_tokenized_target(target_x=target_x, off_target_x=off_target_x, mask=mask, align=align)
        if len(epi.shape) == 1:
            logits, epi_logits, cls_logits, atac_logits = self(masked_target, masked_ot, None, strands)
            epi_loss = torch.zeros(epi_logits.shape, device=epi_logits.device)
        else:
            logits, epi_logits, cls_logits, atac_logits = self(masked_target, masked_ot, epi_masked, strands)
            aligned_epi = self.gather_band(epi[:, self.band_start:self.band_end], align)
            epi_loss = self.epi_loss_fn(epi_logits, aligned_epi) * mask[..., None]
        masked_loss = self.loss_fn(logits.flatten(start_dim=0, end_dim=1), y.flatten()) * mask[..., None].flatten()

        n_supervised = mask.sum().clamp_min(1)
        clsloss = masked_loss.sum() / n_supervised
        epi_loss = epi_loss.sum(dim=(0,1)) / n_supervised

        if self.training:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("lr", lr, prog_bar=True, on_step=False, on_epoch=True, rank_zero_only=True)
        epi_loss = (epi_loss * (self.alpha ** 2)).sum()

        loss = clsloss  # + epi_loss

        if self.use_energy:
            # The R-loop energy model of CRISPRoff is defined only for ungapped
            # guide-target pairs, so the hybrid-energy objective is supervised
            # on the ungapped subset and normalised over it, per batch. The mask
            # is not a model input, so the model cannot condition on it.
            per_sample = self.energy_loss_fct(cls_logits.squeeze(-1), counts.to(torch.float))
            energy_loss = (per_sample * gapless).sum() / gapless.sum().clamp_min(1)
            if self.training:
                self.log("energy_loss", energy_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("energy_supervised_frac", gapless.mean(), on_step=False, on_epoch=True)
            loss = loss + energy_loss

        # ATAC regression: predict mean ATAC signal over the PAM-anchored band.
        # atac_logits: [B, num_atac] — from mean-pooled hidden states
        # atac_true:   [B, num_atac] — mean of the per-nucleotide ATAC values.
        # The band is a fixed genomic interval, so this stays well-defined for a
        # bulged sample: the ATAC track is indexed by coordinate, and the window
        # handed to the model is contiguous and ungapped by construction.
        atac_loss = torch.tensor(0.0, device=loss.device)
        if self.atac_head is not None and isinstance(atac, torch.Tensor) and atac.shape[-1] > 0:
            atac_true = atac[:, self.band_start:self.band_end].mean(dim=1).to(atac_logits.dtype)
            atac_loss = self.atac_loss_fn(atac_logits, atac_true)
            loss = loss + self.atac_weight * atac_loss

        return loss, logits, epi_logits, y, mask, clsloss, epi_loss, atac_loss

    
    def training_step(self, batch, batch_idx):
        loss, logits, epi_logits, y, mask, clsloss, epiloss, atac_loss = self.general_step(batch)
        if self.current_epoch == 0 and batch_idx == 0:
            print(
                f"[TRAIN] first batch ran successfully -> "
                f"loss={loss.item():.4f} cls_loss={clsloss.item():.4f} "
                f"epi_loss={epiloss.item():.4f} atac_loss={atac_loss.item():.4f} "
                f"masked_frac={mask.float().mean().item():.3f}"
            )
            if torch.isnan(loss):
                print("[WARNING] loss is NaN on the very first batch — check data/model setup!")
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_cls_loss", clsloss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_epi_loss", epiloss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_atac_loss", atac_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("epi_logits_var", epi_logits[mask].std(), on_step=False, on_epoch=True)
        preds = torch.softmax(logits, dim=-1)
        self.train_auprc.update(preds[mask], y.int()[mask])
        return loss

    
    def on_train_epoch_end(self):
        # Compute AUPRC after whole epoch
        auprc_val = self.train_auprc.compute()
        self.log("training_auprc", auprc_val, prog_bar=True, sync_dist=True)
        self.train_auprc.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01, betas=(0.9, 0.999))

        # Total number of training steps
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = min(2000, total_steps // 5)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                lr =  float(current_step) / float(max(1, warmup_steps))
            else:
                # After warmup, use cosine decay to zero
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                lr = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

            return lr

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda),
            "interval": "step",  # update per step
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    


def _parse_bulge_profile(profile):
    """Accept a bulge profile written the way JSON can express it.

    JSON object keys must be strings, so ``{"1,0": 0.12}`` is the on-disk form
    of ``{(1, 0): 0.12}``. A dict that already uses tuple keys is passed
    through, as is ``None``.

    Args:
        profile: ``None``, ``{(b_D, b_R): weight}``, or ``{"b_D,b_R": weight}``.

    Returns:
        ``None`` or a dict with tuple keys.
    """
    if profile is None:
        return None
    out = {}
    for key, weight in profile.items():
        if isinstance(key, str):
            b_D, b_R = (int(v) for v in key.split(","))
        else:
            b_D, b_R = key
        out[(int(b_D), int(b_R))] = float(weight)
    return out


def run_pretraining(config):
    batch_size = config["batch_size"]
    hidden_dim = config["hidden_dim"]
    embed_size = config["embed_size"]
    dropout= config["dropout"]
    epi_features = config["epi_features"]
    neighborhood_layers = config["context_layers"]
    seed = config["seed"]
    lr = config["lr"]
    num_epi = sum(EPI_FEATURES[key][-1] for key in epi_features) if epi_features else 0
    epi_weights = 0
    config["num_epi"] = num_epi
    patience = config["patience"]
    train_test_split = config["split"]
    windowsize = config["windowsize"]
    merge = config["merge"]
    model_type = config["model_type"]
    atac_features = config.get("atac_features", [])
    num_atac = len(atac_features)
    atac_weight = config.get("atac_weight", 0.1)

    # --- bulge settings ----------------------------------------------------
    # Omit every one of these and the run is bit-for-bit the original
    # mismatch-only pretraining. "bulge_rate" is the knob for the
    # 0/5/20/50/100 % ablation sweep; a JSON config cannot express tuple keys,
    # so "bulge_profile" accepts {"b_D,b_R": weight} strings as well.
    band_delta = config.get("band_delta", 0)
    bulge_rate = config.get("bulge_rate", None)
    bulge_profile = _parse_bulge_profile(config.get("bulge_profile", None))
    bulge_kwargs = dict(
        band_delta=band_delta,
        bulge_profile=bulge_profile,
        bulge_rate=bulge_rate,
        require_pam=config.get("require_pam", False),
        mismatch_span=config.get("mismatch_span", "site"),
        mismatch_base_factor=config.get("mismatch_base_factor", 3),
        position_tilt=config.get("position_tilt", 0.0),
        mismatch_tilt=config.get("mismatch_tilt", 0.0),
        canonicalise_homopolymers=config.get("canonicalise_homopolymers", True),
        emit_decoy=config.get("emit_decoy", False),
    )

    print(f"[CONFIG] batch_size={batch_size}, windowsize={windowsize}, seed={seed}, lr={lr}")
    print(
        f"[CONFIG] band_delta={band_delta}, bulge_rate={bulge_rate}, "
        f"bulge_profile={'default' if bulge_profile is None else bulge_profile}"
    )
    print(f"[CONFIG] epi_features ({len(epi_features)}): {epi_features}")
    print(f"[CONFIG] num_epi={num_epi}, atac_features={atac_features}, num_atac={num_atac}")
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        print(f"[ENV] CUDA available: True, device_count: {torch.cuda.device_count()}, gpus: {gpu_names}")
    else:
        print("[ENV] CUDA available: False")

    pl.seed_everything(seed,workers=True)

    bw_dir = "EX_BigWigs" if "bw_dir" not in config else config["bw_dir"]
    epi_mode = "bw" if "epi_mode" not in config else config["epi_mode"]
    print(f"[DATA] Building GenomicDataModule (bw_dir={bw_dir}, mode={epi_mode})...")
    num_workers = config.get("num_workers", 20)
    norm_num_samples = config.get("norm_num_samples", 10000)
    dm = GenomicDataModule(
        fasta_path="data/GRCh38.primary_assembly.genome.fa",
        bw_dir=bw_dir,
        epi_features=epi_features,
        window_size=config["windowsize"],
        batch_size=config["batch_size"],
        num_workers=num_workers,
        num_samples=1000000,
        norm_epi=True if config["num_epi"] else False,
        use_energy=config["use_energy"],
        mode=epi_mode,
        atac_features=atac_features,
        norm_num_samples=norm_num_samples,
        **bulge_kwargs,
    )
    model = PreTrainModel(
        context_layers=neighborhood_layers,
        hidden_dim=hidden_dim,
        num_epi=num_epi,
        dropout=dropout,
        lr=lr,
        seed=seed,
        windowsize=windowsize,
        merge=merge,
        epi_weights=epi_weights,
        use_energy=config["use_energy"],
        num_atac=num_atac,
        atac_weight=atac_weight,
        band_delta=band_delta,
    )
    print(f"[MODEL] Built PreTrainModel with {model.hparams.n_trainable_params:,} trainable parameters")

    checkpoint_cb = ModelCheckpoint(
        monitor="train_loss", 
        mode="min", 
        save_top_k=1, 
        filename="best_model",
        save_last=True,
    )

    logger = get_logger(config=config)

    max_steps = config.get("max_steps", 15000)
    accumulate_grad_batches = config.get("accumulate_grad_batches", 25)
    earlystop_cb = EarlyStopping(monitor="train_loss", mode="min", patience=patience)
    trainer = pl.Trainer(
        max_steps=max_steps,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        callbacks=[checkpoint_cb, earlystop_cb],
        log_every_n_steps=10,
        logger=logger,
        deterministic=True,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=0.5,
        precision="bf16-mixed",
        strategy=DDPStrategy(find_unused_parameters=True),
            )
    print(f"[TRAIN] Starting trainer.fit() ... (max_steps={max_steps}, accumulate_grad_batches={accumulate_grad_batches}, num_workers={num_workers})")
    trainer.fit(model, dm, ckpt_path=config["chkpt"] if "chkpt" in config else None)
    print("[TRAIN] trainer.fit() finished.")






if __name__ == "__main__":
    #with open("artificial_param_combinations.json") as handle:
    #    config = json.load(handle)
    idx = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if idx is None:
        # TODO: restore the 7 "EX_"-prefixed features once AGTensorsEFO:0002067 is downloaded.
        # Trimmed to just what's been fully downloaded so far under AGTensorsCL:0000624
        # (ATAC + CHIP_HISTONE) so the batch_size=1 sanity check can run today.
        epi_features = [
            "H3K27ac",
            "H3K27me3",
            "H3K36me3",
            "H3K4me1",
            "H3K4me3",
            "H3K9me3",
        ]
        #epi_features = []
        params = {
            "batch_size": 256,
            "context_layers": 3,
            "hidden_dim": 512,
            "embed_size": 32,
            "dropout": 0.2,
            "epi_features": epi_features,
            "lr": 1e-4,
            "patience": 100000,
            "seed": 42 * 0,
            "split": 1,
            "experiment": "PretrainingArtificialTest",
            "regression": False,
            "windowsize": 512,
            "merge": "early",
            "model_type": "crosscrispr",
            "use_energy": False,
            # TODO: add "AGTensorsEFO:0002067" back once it's downloaded and the "EX_"-prefixed
            # epi_features above are restored -- it's omitted for now since Datasets.py's
            # prepare_data() requires every epi_feature to exist in every listed bw_dir.
            "bw_dir": ["AGTensorsCL:0000624"],
            "epi_mode": "np",
            "atac_features": ["ATAC"],
            "atac_weight": 0.1,
            # num_workers, norm_num_samples, max_steps intentionally omitted here --
            # they fall back to production defaults (20, 10000, 15000).
            # accumulate_grad_batches halved from the 1-GPU default (25 -> 12) because this
            # run uses 2 GPUs: effective batch/step = batch_size * num_gpus * accumulate_grad_batches
            # = 1024 * 2 * 12 = 24,576, close to the original single-GPU 1024*1*25 = 25,600.
            "accumulate_grad_batches": 48,
            #"chkpt": "RUNlogs/PretrainingArtificial/test_split0/ctl6_bs512_ws512_ue20_seed0_hashe0e76e6bafdf121cbfc3/run_/vv6/checkpoints/best_model.ckpt"
        }
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config",
            required=True,
            help="Path to run_configs JSON file"
        )
        args = parser.parse_args()
        with open(args.config) as handle:
            config = json.load(handle)
        idx = int(idx)
        params = config[idx]
    run_pretraining(params)
    
