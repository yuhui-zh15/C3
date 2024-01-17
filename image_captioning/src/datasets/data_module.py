import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader

from .. import builder, enums
from ..parse_data import splits


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.dataset = builder.build_dataset(cfg)

    def train_dataloader(self):
        split = self.cfg.data.train_split
        dataset = self.dataset(self.cfg, split=split)

        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def val_dataloader(self):
        dataset = self.dataset(self.cfg, split="val")
        bs = 1 if self.cfg.val_eval else self.cfg.train.batch_size
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            batch_size=bs,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        if self.cfg.test_split == "all":
            datasets = [self.dataset(self.cfg, split=split) for split in splits]
            dataset = ConcatDataset(datasets)
        elif self.cfg.test_split == "train+restval":
            train_dataset = self.dataset(self.cfg, split="train")
            restval_dataset = self.dataset(self.cfg, split="restval")
            dataset = ConcatDataset([train_dataset, restval_dataset])
        else:
            dataset = self.dataset(self.cfg, split=self.cfg.test_split)

        if self.cfg.decoder.modality == enums.Modality.Language:
            batch_size = 1
        else:
            batch_size = self.cfg.train.batch_size

        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            batch_size=batch_size,
            num_workers=1,
        )

    def all_dataloader(self):
        datasets = [self.dataset(self.cfg, split=split) for split in splits]
        dataset = ConcatDataset(datasets)

        dataset = self.dataset(self.cfg, split=self.cfg.test_split)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )
