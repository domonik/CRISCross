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
    epi_hash = short_hash(config["epi_features"], config["num_epi"])
    base_dir = f"RUNlogs/{config['experiment']}/test_split{config['split']}/ctl{config['context_layers']}_bs{config['batch_size']}_ws{config["windowsize"]}_ue{config['num_epi']}_seed{config['seed']}_energy{config["use_energy"]}_hash{epi_hash}" 
    existing = os.listdir(os.path.join(base_dir, "run_")) if os.path.exists(base_dir) else []
    print(f"BASE DIR: {base_dir}")
    version = f"v{len(existing)}"
    logger = TensorBoardLogger(
        save_dir=base_dir,   # your custom folder
        name=f"run_",
        version=f"v{version}"
    )   
    return logger

class PreTrainModel(pl.LightningModule):
    def __init__(self, context_layers, hidden_dim, num_epi, dropout, seed, windowsize, merge, epi_weights, lr=1e-4, borders=None, use_energy=False, num_atac=0, atac_weight=0.1):
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
            merge=merge

        )
        self.windowsize = windowsize
        
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
        self.energy_loss_fct = nn.MSELoss()
        self.per_nt_classifier = nn.Linear(hidden_dim, 25)
        self.per_nt_epi_head = nn.Linear(hidden_dim, num_epi)
        self.auprc = MulticlassAveragePrecision(num_classes=25)
        self.train_auprc = MulticlassAveragePrecision(num_classes=25)
        self._first_batches_features = []

        self.num_atac = num_atac
        self.atac_weight = atac_weight
        self.atac_head = nn.Linear(hidden_dim, num_atac) if num_atac > 0 else None
        self.atac_loss_fn = nn.MSELoss()



    
    def forward(self, target_x, off_target_x, epi, strands):
        cls_logits, hidden = self.model(target_x, off_target_x, strands, epi)
        epi_logits = self.per_nt_epi_head(hidden)
        atac_logits = self.atac_head(hidden) if self.atac_head is not None else None
        logits = self.per_nt_classifier(hidden)
        return logits, epi_logits, cls_logits, atac_logits
    
    def mask_shit(self, batch):
        target_x, off_target_x, epi, y, counts, strands, atac = batch
        batch_size, seq_len = target_x.shape
        center = off_target_x.shape[1] // 2 + off_target_x.shape[1] % 2
        target_x = target_x.clone()


        # Generate a random mask
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

        off_target_x = off_target_x.clone()
        off_target_x[:, center - 23//2 - 1:center+23//2] = off_target_x[:, center - 23//2 - 1:center+23//2].masked_fill(mask_mask, 0)


        epi = epi.clone()
        if len(epi.shape) > 1:
            epi[:, center - 23//2 - 1:center+23//2] = epi[:, center - 23//2 - 1:center+23//2].masked_fill(mask_tensor.unsqueeze(-1), 0)
            d = torch.randint(low=0, high=min(32, self.windowsize // 2), size=(1,))
            B, T = epi.shape[0:2]
            if self.extra_epi_mask:
                d = torch.randint(min(self.windowsize // 2, 128 // 2), min(128 // 2, self.windowsize // 2)+1, (B,), device=epi.device)
                idx = torch.arange(T, device=epi.device).unsqueeze(0) 
                epi_mask = (idx >= (center - d).unsqueeze(1)) & (idx < (center + d).unsqueeze(1))

                epi[epi_mask] = 0
        else:
            epi_mask = torch.zeros(epi.shape)
        return target_x, off_target_x, epi, mask_tensor
        

    def compute_tokenized_target(self, target_x, off_target_x, mask):
        bs, slen = target_x.shape
        center = off_target_x.shape[1] // 2 + off_target_x.shape[1] % 2

        centered_off_target_x = off_target_x[:, center - 23//2 - 1:center+23//2].clone()


        y1 = torch.zeros((bs, slen), dtype=torch.long).to(target_x.device)
        y2 = torch.zeros((bs, slen), dtype=torch.long).to(target_x.device)
        y1[mask] = target_x[mask].to(torch.long)
        y2[mask] = centered_off_target_x[mask].to(torch.long)

        #y1[y1 > 1] -= 2
        #y2[y2 > 1] -= 2
        y = y1 * self.max_idx + y2
        return y




    def general_step(self, batch):
        target_x, off_target_x, epi, y, counts, strands, atac = batch
        masked_target, masked_ot, epi_masked, mask = self.mask_shit(batch)
        center = off_target_x.shape[1] // 2 + off_target_x.shape[1] % 2
        assert (off_target_x[:, center - 23//2 - 1:center+23//2] == target_x).sum(axis=1).min() >= 15

        bs, slen = target_x.shape
        y = self.compute_tokenized_target(target_x=target_x, off_target_x=off_target_x, mask=mask)
        if len(epi.shape) == 1:
            logits, epi_logits, cls_logits, atac_logits = self(masked_target, masked_ot, None, strands)
            epi_loss = torch.zeros(epi_logits.shape, device=epi_logits.device)
        else:
            logits, epi_logits, cls_logits, atac_logits = self(masked_target, masked_ot, epi_masked, strands)
            epi_loss = self.epi_loss_fn(epi_logits, epi[:, center - 23//2 - 1:center+23//2]) * mask[..., None]
        masked_loss = self.loss_fn(logits.flatten(start_dim=0, end_dim=1), y.flatten()) * mask[..., None].flatten()

        clsloss = masked_loss.sum() / mask.sum()
        epi_loss = epi_loss.sum(dim=(0,1)) / mask.sum()

        if self.training:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("lr", lr, prog_bar=True, on_step=False, on_epoch=True, rank_zero_only=True)
        epi_loss = (epi_loss * (self.alpha ** 2)).sum()

        loss = clsloss  # + epi_loss

        if self.use_energy:
            energy_loss = self.energy_loss_fct(cls_logits.squeeze(), counts.to(torch.float))
            if self.training:
                self.log("energy_loss", energy_loss, on_step=False, on_epoch=True, prog_bar=True)
            loss = loss + energy_loss

        # ATAC regression: predict per-nucleotide ATAC signal at the center 23nt window
        atac_loss = torch.tensor(0.0, device=loss.device)
        if self.atac_head is not None and isinstance(atac, torch.Tensor) and atac.shape[-1] > 0:
            # atac_logits: [B, 23, num_atac] — per-nt predictions from hidden states
            # atac target: slice the center 23nt from the full window
            atac_true = atac[:, center - 23//2 - 1:center + 23//2].to(atac_logits.dtype)
            atac_loss = self.atac_loss_fn(atac_logits, atac_true)
            loss = loss + self.atac_weight * atac_loss

        return loss, logits, epi_logits, y, mask, clsloss, epi_loss, atac_loss

    
    def training_step(self, batch, batch_idx):
        loss, logits, epi_logits, y, mask, clsloss, epiloss, atac_loss = self.general_step(batch)
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

    pl.seed_everything(seed,workers=True)

    bw_dir = "EX_BigWigs" if "bw_dir" not in config else config["bw_dir"]
    epi_mode = "bw" if "epi_mode" not in config else config["epi_mode"]
    dm = GenomicDataModule(
        fasta_path="GRCh38.primary_assembly.genome.fa",
        bw_dir=bw_dir,
        epi_features=epi_features,
        window_size=config["windowsize"],
        batch_size=config["batch_size"],
        num_workers=20,
        num_samples=1000000,
        norm_epi=True if config["num_epi"] else False,
        use_energy=config["use_energy"],
        mode=epi_mode,
        atac_features=atac_features,
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
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="train_loss", 
        mode="min", 
        save_top_k=1, 
        filename="best_model",
        save_last=True,
    )

    logger = get_logger(config=config)

    earlystop_cb = EarlyStopping(monitor="train_loss", mode="min", patience=patience)
    trainer = pl.Trainer(
        max_steps=15000,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count(),
        callbacks=[checkpoint_cb, earlystop_cb],
        log_every_n_steps=10,
        logger=logger,
        deterministic=True,
        accumulate_grad_batches=25,
        gradient_clip_val=0.5,
        precision="bf16-mixed",
        strategy=DDPStrategy(find_unused_parameters=True),
            )
    trainer.fit(model, dm, ckpt_path=config["chkpt"] if "chkpt" in config else None)






if __name__ == "__main__":
    #with open("artificial_param_combinations.json") as handle:
    #    config = json.load(handle)
    idx = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if idx is None:
        epi_features = [            
            "H3K4me1",  
            "H3K4me3",  
            "H3K36me3",
        ]
        #epi_features = []
        params = {
            "batch_size": 1024,
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
            "windowsize": 23,
            "merge": None,
            "model_type": "crosscrispr",
            "use_energy": False,
            "bw_dir": ["AGTensorsCL:0000624", "AGTensorsEFO:0002067"],
            "epi_mode": "np",
            "atac_features": ["ATAC"],
            "atac_weight": 0.1,
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
    