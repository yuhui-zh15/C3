import os

import argparse
import torch
import datetime
import math
import random

from src import utils, builder

from dateutil import tz
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

DATA_PREFIX = "."


def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="paths to base config")
    parser.add_argument(
        "--train", action="store_true", default=False, help="specify to train model"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="specify to debug model"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="specify to test model"
        "By default run.py trains a model based on config file",
    )
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=None,
        help="fraction of training set to sample",
    )
    parser.add_argument(
        "--remove_mean",
        action="store_true",
        help="whether or not to mean from the input embed",
    )
    parser.add_argument(
        "--normalize_prefix",
        action="store_true",
        default=None,
        help="whether to normalize clip embeds or not at the beginning (should be redundant)",
    )

    parser.add_argument(
        "--re_normalize_prefix",
        action="store_true",
        default=None,
        help="whether to normalize clip embeds or not at the very end",
    )
    
    parser.add_argument(
        "--add_gaussian_noise",
        action="store_true",
        default=None,
        help="whether to add gaussian noise to input embeds",
    )

    parser.add_argument(
        "--noise_level",
        default=math.sqrt(0.016),
        type=float,
        help="noise level",
    )
    parser.add_argument("--lr", type=float, default=None, help="learning rate")
    parser.add_argument("--test_split", type=str, default="test", help="test split")
    parser.add_argument("--random_seed", type=int, default=1234, help="random seed")

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output")

    parser.add_argument("--val_eval", action="store_true", help='whether to run evaluation metrics over the validation set after every training epoch')
    parser.add_argument("--cross_modal_val", action="store_true", help='whether to run cross-modal evaluation over validation after every training epoch')

    parser.add_argument("--subsample_val_test", action="store_true")

    # parser = Trainer.add_argparse_args(parser)

    args, unknown = parser.parse_known_args()
    cli = [u.strip("--") for u in unknown]  # remove strings leading to flag

    # add command line argments to config
    cfg = OmegaConf.load(args.config)
    cli = OmegaConf.from_dotlist(cli)
    cli_flat = utils.flatten(cli)
    cfg.hyperparameters = cli_flat  # hyperparameter defaults

    args.gpus = torch.cuda.device_count()
    if args.gpus is not None:
        cfg.lightning.trainer.gpus = str(args.gpus)

    cfg.val_eval = args.val_eval
    cfg.cross_modal_val = args.cross_modal_val

    cfg.checkpoint = args.checkpoint

    cfg.model.normalize_prefix = False
    if args.normalize_prefix is not None:
        cfg.model.normalize_prefix = args.normalize_prefix
        cfg.experiment_name += "_normed"

    cfg.experiment_name += f"_{cfg.model.mapping_type}"

    cfg.data.sample_frac = args.sample_frac
    if not OmegaConf.is_none(cfg.data, "sample_frac"):
        cfg.experiment_name += f"_frac{args.sample_frac}"

    cfg.data.remove_mean = args.remove_mean
    if args.remove_mean:
        cfg.experiment_name += f"_remove_mean"
        
    cfg.data.add_gaussian_noise = args.add_gaussian_noise
    if args.add_gaussian_noise:
        cfg.experiment_name += f"_add_gaussian_noise_level_{round(args.noise_level, 5)}"
    
    cfg.model.re_normalize_prefix = args.re_normalize_prefix
    if not args.re_normalize_prefix:
        cfg.experiment_name += "_not_renormed"
    
    cfg.model.subsample_val_test = args.subsample_val_test
    if args.subsample_val_test:
        cfg.experiment_name += "_subsample_val_test"
    
    cfg.experiment_name += f"_{cfg.train.optimizer.name}"

    if args.lr:
        cfg.lightning.trainer.lr = args.lr
        cfg.experiment_name += f"_lr_{args.lr}"
        
    cfg.test_split = args.test_split
    cfg.noise_level = args.noise_level

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    cfg.extension = timestamp

    # check debug
    if args.debug:
        cfg.train.num_workers = 0

    cfg.seed = args.random_seed
    seed_everything(args.random_seed)

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if cfg.checkpoint is None and not OmegaConf.is_none(cfg.model, "pretrain_ckpt"):
        cfg.checkpoint = cfg.model.pretrain_ckpt
    if cfg.checkpoint and "noise" in cfg.checkpoint:
        cfg.experiment_name += f"_add_noise_in_pretrain"
        
    cfg.experiment_name += f"_seed_{args.random_seed}"
        

    return cfg, args


def create_directories(cfg):
    # set directory names
    cfg.output_dir = f"{DATA_PREFIX}/data/output/{cfg.experiment_name}/{cfg.extension}"
    cfg.lightning.logger.name = f"{cfg.experiment_name}/{cfg.extension}"
    cfg.lightning.checkpoint_callback.dirpath = f"{DATA_PREFIX}/data/c3/{cfg.data.dataset}/ckpt/{cfg.experiment_name}/{cfg.extension}"

    # create directories
    if not os.path.exists(cfg.lightning.logger.save_dir):
        os.makedirs(cfg.lightning.logger.save_dir)
    if not os.path.exists(cfg.lightning.checkpoint_callback.dirpath):
        print(cfg.lightning.checkpoint_callback.dirpath)
        os.makedirs(cfg.lightning.checkpoint_callback.dirpath)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    return cfg


def setup(cfg, test_split=False):
    # create output, logger and ckpt directories if split != test
    if not test_split:
        cfg = create_directories(cfg)

        # logging
        loggers = []
        if "logger" in cfg.lightning:
            logger_type = cfg.lightning.logger.pop("logger_type")
            logger_class = getattr(pl_loggers, logger_type)
            logger = logger_class(**cfg.lightning.logger)
            loggers.append(logger)
            cfg.lightning.logger.logger_type = logger_type

        # callbacks
        callbacks = [LearningRateMonitor(logging_interval="step")]
        if "checkpoint_callback" in cfg.lightning:
            checkpoint_callback = ModelCheckpoint(**cfg.lightning.checkpoint_callback)
            callbacks.append(checkpoint_callback)
        if "early_stopping_callback" in cfg.lightning:
            early_stopping_callback = EarlyStopping(
                **cfg.lightning.early_stopping_callback
            )
            callbacks.append(early_stopping_callback)

        # save config
        config_path = os.path.join(cfg.output_dir, "config.yaml")
        config_path_ckpt = os.path.join(
            cfg.lightning.checkpoint_callback.dirpath, "config.yaml"
        )
        with open(config_path, "w") as fp:
            OmegaConf.save(config=cfg, f=fp.name)
        with open(config_path_ckpt, "w") as fp:
            OmegaConf.save(config=cfg, f=fp.name)

    else:
        loggers = []
        callbacks = []
        checkpoint_callback = None

    # get datamodule
    dm = builder.build_data_module(cfg)
    cfg.data.num_batches = len(dm.train_dataloader())

    # reduce number of warmup steps for small fractions to keep ratio of num warmup steps to
    # num batches in a single epoch the same
    if not OmegaConf.is_none(cfg.data, "sample_frac") and not OmegaConf.is_none(
        cfg.train, "scheduler"
    ):
        cfg.train.scheduler.warmup_steps = int(
            cfg.train.scheduler.warmup_steps * cfg.data.sample_frac
        )

    # define lightning module
    model = builder.build_lightning_model(cfg)

    # setup pytorch-lightning trainer
    lr = cfg.lightning.trainer.pop('lr')
    cfg.lightning.trainer.pop('gpus')
    trainer = Trainer(
        **cfg.lightning.trainer, deterministic=False, callbacks=callbacks, logger=loggers
    )  # note: determinstic is set to True in eval/predict.py with warn_only=True
    
    cfg.lightning.trainer.lr = lr

    return trainer, model, dm, checkpoint_callback


if __name__ == "__main__":
    cfg, args = parse_configs()

    # Remove pre-trained sub-model checkpoint if full model checkpoint provided
    if not OmegaConf.is_none(cfg, "checkpoint"):
        if not OmegaConf.is_none(cfg.decoder, "checkpoint"):
            cfg.decoder.pop("checkpoint")

        if not OmegaConf.is_none(cfg.model, "checkpoint"):
            cfg.model.pop("checkpoint")

    if args.train:
        trainer, model, dm, checkpoint_callback = setup(cfg)
        trainer.fit(model, dm)
        best_ckpt = checkpoint_callback.best_model_path
        cfg.checkpoint = best_ckpt
        print(f"Best checkpoint path: {best_ckpt}")

    if args.test:
        # assert(not OmegaConf.is_none(cfg, "checkpoint"), "cfg.checkpoint cannot be None!")

        if not OmegaConf.is_none(cfg, "checkpoint"):
            print("=" * 80)
            print(cfg.checkpoint)
            print("=" * 80)

            # Remove pre-trained model checkpoint
            if not OmegaConf.is_none(cfg.decoder, "checkpoint"):
                cfg.decoder.pop("checkpoint")

            if not OmegaConf.is_none(cfg.model, "checkpoint"):
                cfg.model.pop("checkpoint")

            # Change train_split (unused anyway)
            cfg.data.train_split = "val"

            cfg.output_dir = "/".join(cfg.checkpoint.split("/")[:-1]).replace(
                "ckpt", "output"
            )
        else:
            cfg.output_dir = args.output_dir
            os.makedirs(cfg.output_dir, exist_ok=True)
        print(f"Output dir: {cfg.output_dir}")
        trainer, model, dm, checkpoint_callback = setup(cfg, test_split=True)
        trainer.test(model=model, datamodule=dm)
