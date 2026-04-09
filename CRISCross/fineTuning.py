import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import BinaryAveragePrecision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
import pandas as pd
import torch.multiprocessing as mp
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
import multiprocessing.util
import collections
from typing import Dict, List, Tuple
from CRISCross.models import CRISCross
from CRISCross.Datasets import MyDataModule, EPI_FEATURES, GenomicDataModule
import json
from torchmetrics import Metric
from torchmetrics.functional import spearman_corrcoef
from CRISCross.pretrainArtificial import PreTrainModel, get_logger

from torch.optim.lr_scheduler import LambdaLR


class SpearmanCorr(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds, targets):
        self.preds.append(preds.detach())
        self.targets.append(targets.detach())

    def compute(self):
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        return spearman_corrcoef(preds, targets)



class PLCRISPRWrapper(pl.LightningModule):
    def __init__(self, model_type, embed_size, context_layers, hidden_dim, num_epi, dropout, seed, windowsize, merge, lr=1e-4, borders = None):
        super().__init__()
        
        if borders is not None:
            raise NotImplementedError()
            self.criterion = BarDistributionConfig(full_support=True, borders=borders).get_criterion()
            self.output_size = len(borders) - 1
            self.regression = True
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")
            self.output_size = 1
            self.regression = False



        self.model = CRISCross(
            vocab_size=5,
            dropout=dropout,
            context_layers=context_layers,
            hidden_dim=hidden_dim,
            num_epi=num_epi,
            output_size=self.output_size,
            windowsize=windowsize,
            merge=merge
        )
        
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.hparams.n_trainable_params = n_params
        self.hparams.seed = seed
        self.save_hyperparameters()

        self.lr = lr
        self.auprc = BinaryAveragePrecision()  # torchmetrics AUPRC
        self.test_auprc = BinaryAveragePrecision()
        self.train_spearman = SpearmanCorr()
        self.val_spearman = SpearmanCorr()
        self.test_spearman = SpearmanCorr()
    
    def forward(self, target_x, off_target_x, epi, strands):
        return self.model(target_x, off_target_x, strands, epi)

    def general_step(self, batch):
            
        target_x, off_target_x, epi, y, counts, strands = batch
        center = off_target_x.shape[1] // 2 + off_target_x.shape[1] % 2
        t_vals = (off_target_x[:, center - 23//2 - 1:center+23//2] == target_x).sum(axis=1)
        mval = t_vals.min()
        assert mval >= 15, f"Min of {mval} detected"


        res = self(target_x, off_target_x, epi, strands)
        logits = res[0] if isinstance(res, tuple) else res
        logits = logits.squeeze(1)


        if self.regression:
            loss = self.criterion(logits[None, ...], counts.float()).mean()
            preds = self.criterion.mean(logits)
        else:
           
            loss = self.criterion(logits, y.float()).mean()
            preds = torch.sigmoid(logits)
        return loss, preds




    
    def training_step(self, batch, batch_idx):
        loss, preds = self.general_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        y, counts, strands = batch[3:]
        loss, preds = self.general_step(batch)
        
        # Update AUPRC metric
        self.auprc.update(preds, y.int())
        if self.regression:
            preds = preds[y == 1]
            counts = counts[y == 1]
            self.val_spearman.update(preds, counts)

        
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        y, counts, strands = batch[3:]
        loss, preds = self.general_step(batch)
        
        # Update AUPRC metric
        self.test_auprc.update(preds, y.int())
        if self.regression:
            preds = preds[y == 1]
            counts = counts[y == 1]

            self.test_spearman.update(preds, counts)
        # Update test AUPRC metric
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        if self.regression:
            spearman = self.test_spearman.compute()
            self.log("test_spearman", spearman, prog_bar=True)
            self.test_spearman.reset()
        auprc_val = self.test_auprc.compute()
        self.log("test_auprc", auprc_val, prog_bar=True)
        self.auprc.reset()
    
    
    def on_validation_epoch_end(self):
        if self.regression:
            spearman = self.val_spearman.compute()
            self.log("val_spearman", spearman, prog_bar=True)
            # Reset for safety
            self.val_spearman.reset()
        auprc_val = self.auprc.compute()
        self.log("val_auprc", auprc_val, prog_bar=True)
        self.auprc.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01, betas=(0.9, 0.999))

        # Total number of training steps
        train_loader = self.trainer.datamodule.train_dataloader()
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = min(10, total_steps // 5)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda),
            "interval": "step",  # update per step
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def get_distribution_borders(datamodule):
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    ys = []
    for idx, batch in enumerate(train_loader):
        target_x, off_target_x, epi, y, counts, strands = batch      
        if idx == 100:
            break
        ys.append(counts)
    ys = torch.concat(ys)
    borders = [0.] + bar_distribution.get_bucket_borders(num_outputs=15, ys=ys).tolist() + [1.0]
    borders = torch.tensor(borders).unique().tolist()
    print(f"Using these borders {borders}")
    return borders




class LastKCheckpoint(ModelCheckpoint):
    def __init__(self, k=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self._recent = []

    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)

        self._recent.append(filepath)

        if len(self._recent) > self.k:
            old = self._recent.pop(0)
            if os.path.exists(old):
                os.remove(old)

from pathlib import Path

def save_average_checkpoint_lightning(model, ckpt_paths, trainer, output_name="average_model.ckpt"):
    if not ckpt_paths:
        raise ValueError("No checkpoints provided")

    folder = os.path.dirname(ckpt_paths[0])
    output_path = os.path.join(folder, output_name)

    # Load first checkpoint to get hparams
    first_ckpt = torch.load(ckpt_paths[0], map_location="cpu")
    hparams = first_ckpt.get("hyper_parameters", first_ckpt.get("hparams", {}))

    # Instantiate the module with the same hparams

    # Average floating-point weights
    avg_state = None
    n = len(ckpt_paths)
    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]
        if avg_state is None:
            avg_state = {k: v.clone() for k, v in state_dict.items()}
        else:
            for k in avg_state:
                if avg_state[k].dtype.is_floating_point:
                    avg_state[k] += state_dict[k]

    for k in avg_state:
        if avg_state[k].dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            avg_state[k] /= n

    # Load averaged weights
    model.load_state_dict(avg_state)

    # Save using Lightning function via trainer
    trainer.save_checkpoint(output_path)

    return output_path


def run_training(config):
    batch_size = config["batch_size"]
    hidden_dim = config["hidden_dim"]
    embed_size = config["embed_size"]
    dropout= config["dropout"]
    epi_features = config["epi_features"]
    context_layers = config["context_layers"]
    seeds = config["seed"]
    lr = config["lr"]
    num_epi = sum(EPI_FEATURES[key][-1] for key in epi_features)
    config["num_epi"] = num_epi
    patience = config["patience"]
    train_test_split = config["split"]
    windowsize = config["windowsize"]
    merge = config["merge"]
    model_type = config["model_type"]
    epi_mode = config["epi_mode"]
    bw_dir = config["bw_dir"]
    run_settings = pd.read_csv(config["run_settings"], sep="\t")
    run_settings["val_set"] = run_settings["val_set"].apply(
        lambda x: eval(x) if not pd.isna(x) else None
    )
    run_settings["test_set"] = run_settings["test_set"].apply(eval)
    run_settings["exclude"] = run_settings["exclude"].apply(eval)


    if isinstance(seeds, int):
        seeds = [seeds]
    else:
        seeds = config["seed"]
    for seed_idx, seed in enumerate(seeds):
        config["seed"] = seed
        df = pd.read_csv(config["dataset"], sep="\t")



        pl.seed_everything(seed,workers=True)


        dm = GenomicDataModule(
            fasta_path="GRCh38.primary_assembly.genome.fa",
            bw_dir=bw_dir,
            epi_features=epi_features,
            window_size=config["windowsize"],
            batch_size=config["batch_size"],
            num_workers = 10,
            num_samples=batch_size*10,
            norm_epi=True if config["num_epi"] else False,
            use_energy=False,
            mode=epi_mode,
            df=df,
            val_guides=run_settings.iloc[train_test_split]["val_set"], 
            test_guides=run_settings.iloc[train_test_split]["test_set"], 

        )

        if config["regression"]:
            borders = get_distribution_borders(dm)
        else:
            borders = None

        model = PLCRISPRWrapper(
            model_type=model_type,
            embed_size=embed_size,
            context_layers=context_layers,
            hidden_dim=hidden_dim,
            num_epi=num_epi,
            dropout=dropout,
            lr=lr,
            seed=seed,
            borders=borders,
            windowsize=windowsize,
            merge=merge

        )
        if "chkpt" in config:
            if isinstance(config["chkpt"], list):
                cur_chkpt = config["chkpt"][seed_idx]
            else:
                cur_chkpt = config["chkpt"]
            try:
                ptm = PreTrainModel.load_from_checkpoint(cur_chkpt, weights_only=False)
                print("Using pretrained model")
                model.model.load_state_dict(ptm.model.state_dict())
            except TypeError:
                ptm = PLCRISPRWrapper.load_from_checkpoint(cur_chkpt, weights_only=False)
                model.model.load_state_dict(ptm.model.state_dict())

        callbacks = []
        if run_settings.iloc[train_test_split]["val_set"] is not None:
            checkpoint_cb = ModelCheckpoint(
                monitor="val_auprc",
                mode="max",
                save_top_k=1 if "use_top_k" not in config else config["use_top_k"],
                filename="best_model",
                save_last=True
            )
            earlystop_cb = EarlyStopping(monitor="val_auprc", mode="max", patience=patience)
            

            callbacks.append(checkpoint_cb)
            callbacks.append(earlystop_cb)
        else:
            checkpoint_cb = LastKCheckpoint(
                k=1,                      # best + 5 surrounding or just last 6
                filename="epoch-{epoch}",
                every_n_epochs=1,
                save_top_k=-1             # required so Lightning doesn't filter
            )
            callbacks.append(checkpoint_cb)
        swa_callback = StochasticWeightAveraging(
            swa_lrs=config["lr"],             # SWA learning rate
            swa_epoch_start=config["swa_epoch_start"] if   "swa_epoch_start" in config else 10000     # start SWA at 80% of total epochs
        )
        callbacks.append(swa_callback)

        logger = get_logger(config=config)

        trainer = pl.Trainer(
            #max_steps=config["max_steps"] if "max_steps" in config else 1000,
            max_epochs=config["max_epochs"] if "max_epochs" in config else 1000,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=callbacks,
            log_every_n_steps=10,
            logger=logger,
            deterministic="warn",
            gradient_clip_val=0.5,
            accumulate_grad_batches=config["accumulate_grad_batches"] if "accumulate_grad_batches" in config else 1
        )
        



        if config["fit"]:
            trainer.fit(model, dm)
            load = checkpoint_cb.best_model_path
            best_model = PLCRISPRWrapper.load_from_checkpoint(load, weights_only=False)
            if "use_top_k" in config and config["use_top_k"] > 1:
                if run_settings.iloc[train_test_split]["val_set"] is None:
                    ckpt_paths = checkpoint_cb._recent  # last k checkpoints
                else:
                    ckpt_paths = list(checkpoint_cb.best_k_models.keys())
                avg_path = save_average_checkpoint_lightning(best_model, ckpt_paths, trainer)
                print(f"Averaged model saved to: {avg_path}")
                best_model = PLCRISPRWrapper.load_from_checkpoint(avg_path)
                
            swa_model = swa_callback._average_model
            print("SWA n_averaged:", swa_callback.n_averaged)
            if swa_model is not None:
                trainer.model.load_state_dict(swa_model.state_dict())
            ckpt_dir = checkpoint_cb.dirpath
            trainer.save_checkpoint(f"{ckpt_dir}/swa_model.ckpt")
        else:
            best_model = model
            

        if len(run_settings.iloc[train_test_split]["test_set"]):
            test_result = trainer.test(best_model, dm)

if __name__ == "__main__":
    import argparse
    idx = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if idx is None:
        epi_features = [            
            #"EX_ATAC",
            #"EX_H3K4me1",  
            #"EX_H3K4me3",  
            #"EX_H3K9ac", # not present in alphagenome
            #"EX_H3K9me3",  
            #"EX_H3K27ac",  
            #"EX_H3K27me3",  
            #"EX_H3K36me3",
            "ATAC",
            #"DNASE", 
            #"RNA_SEQ",
            "CHIP_HISTONE"
        ]
        #epi_features = []
        epi_features = [
            "ATAC",
            "H3K27ac",
            "H3K27me3",
            "H3K36me3",
            "H3K4me1",
            "H3K4me3",
        ]
        #epi_features = []
        params = {
            "batch_size": 128,
            "context_layers": 3,
            "embed_size": 32,
            "hidden_dim": 512,
            "dropout": 0.2,
            "lr": 0.0001,
            "patience": 20,
            "experiment": "CCDebug/FineTune",
            "regression": False,
            "merge": "early",
            "model_type": "crosscrispr",
            "accumulate_grad_batches": 2,
            "epi_mode": "np",
            "epi_features": [
            "ATAC"
            ],
            "seed": [
            0
            ],
            "split": 0,
            "bw_dir": [
            "AGTensorsCL:0000624",
            "AGTensorsEFO:0002067"
            ],
            "fit": True,
            "windowsize": 512,
            "run_settings": "run_settings/RunSettingsJoinedFineTune_seed0.tsv",
            "dataset": "joined_datasets/T_cell_K562Val_joined_seed0.tsv",
            "IDX": 1,
            "num_epi": 1,
            "chkpt": "RUNlogs/PretrainingArtificialPaperAG2/test_split1/ctl3_bs1024_ws512_ue1_seed0_energyTrue_hash2/run_/vv1/checkpoints/last.ckpt"
        }
    else:
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
    run_training(params)



