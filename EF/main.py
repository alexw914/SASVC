import argparse
import os
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.strategies.ddp import DDPStrategy

from utils.utils import *

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def main(args):
    # load configurations and set seed
    config = OmegaConf.load(args.config)
    output_dir = Path(args.output_dir)
    pl.seed_everything(config.seed, workers=True)

    # generate speaker-utterance meta information
    if not (
        os.path.exists(config.dirs.spk_meta + "spk_meta_trn.pk")
        and os.path.exists(config.dirs.spk_meta + "spk_meta_dev.pk")
        and os.path.exists(config.dirs.spk_meta + "spk_meta_eval.pk")
    ):
        generate_spk_meta(config)

    # configure paths
    model_tag = os.path.splitext(os.path.basename(args.config))[0]
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    model_save_path.mkdir(parents=True, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    _system = import_module("systems.{}".format(config.pl_system))
    _system = getattr(_system, "System")
    system = _system(config)

    # Configure logging and callbacks
    logger = [
        pl.loggers.TensorBoardLogger(save_dir=model_tag, version=1, name="tsbd_logs"),
        pl.loggers.csv_logs.CSVLogger(
            save_dir=model_tag,
            version=1,
            name="csv_logs",
            flush_logs_every_n_steps=config.progbar_refresh * 100,
        ),
    ]

    callbacks = [
        # pl.callbacks.ModelSummary(max_depth=3),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.ModelCheckpoint(
            dirpath=model_save_path,
            filename="{epoch}-{sasv_eer_dev:.5f}",
            monitor="sasv_eer_dev",
            mode="min",
            every_n_epochs=config.val_interval_epoch,
            save_top_k=config.save_top_k,
        ),
    ]

    # Train / Evaluate
    gpus = find_gpus(config.ngpus, min_req_mem=config.min_req_mem)
    if gpus == -1:
        raise ValueError("Required GPUs are not available")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    trainer = pl.Trainer(
        accelerator="gpu",
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        devices=config.ngpus,
        fast_dev_run=config.fast_dev_run,
        gradient_clip_val=config.gradient_clip
        if config.gradient_clip is not None
        else 0,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        logger=logger,
        max_epochs=config.epoch,
        num_sanity_val_steps=0,
        reload_dataloaders_every_n_epochs=config.loader.reload_every_n_epoch
        if config.loader.reload_every_n_epoch is not None
        else config.epoch,
        sync_batchnorm=True,
        val_check_interval=1.0,  # 0.25 validates 4 times every epoch
        strategy = DDPStrategy(find_unused_parameters=True),
    )

    trainer.fit(system)
    trainer.test(ckpt_path="best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SASVC2022 embedding fusion framework.")
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        help="configuration file",
        required=True,
        default="configs/sasv.conf",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    main(parser.parse_args())
