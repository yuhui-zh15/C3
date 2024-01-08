import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from omegaconf import OmegaConf

from . import lightning, datasets, losses


def build_data_module(cfg):
    return datasets.data_module.DataModule(cfg)


def build_dataset(cfg):
    if cfg.data.dataset.lower() in datasets.ALL_DATASETS:
        return datasets.ALL_DATASETS[cfg.data.dataset.lower()]
    else:
        raise NotImplementedError(
            f"Dataset not implemented for {cfg.data.dataset.lower()}"
        )

def build_lightning_model(cfg):
    
    model = lightning.ClipCaptionLightningModel
        
    if OmegaConf.is_none(cfg, "checkpoint"):
        return model(cfg)
    else:
        checkpoint_path = cfg.checkpoint
        print('='*80)
        print(f'*** Loading checkpoint: {checkpoint_path}')
        print('='*80)
        if checkpoint_path == 'skip': 
            return model(cfg)
        return model.load_from_checkpoint(checkpoint_path, cfg=cfg)
        

def build_optimizer(cfg, model):
    params = model.parameters()
    name = cfg.train.optimizer.pop('name')
    if hasattr(torch.optim, name):
        optimizer_fn = getattr(torch.optim, name)
    else:
        raise ValueError(f'torch.optim has no optimizer \'{name}\'.')

    optimizer = optimizer_fn(params, lr=cfg.lightning.trainer.lr, **cfg.train.optimizer)
    return optimizer


def build_scheduler(cfg, optimizer):
    if cfg.train.scheduler.name is not None:
        scheduler_name = cfg.train.scheduler.pop("name")
        
        if scheduler_name == 'linear_schedule_with_warmup':
            scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.train.scheduler.warmup_steps, 
                num_training_steps=cfg.lightning.trainer.max_epochs * cfg.data.num_batches)
    return scheduler


def build_loss(cfg):
    # get loss function
    args = cfg.train.loss_fn
    name = cfg.train.loss_fn.pop('name')
    
    if hasattr(torch.nn, name):
        loss_fn = getattr(nn, name)(**args)
    else:
        loss_fn = losses.__dict__[name]
    
    cfg.train.loss_fn.name = name
    return loss_fn

