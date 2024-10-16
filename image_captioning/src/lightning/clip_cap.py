import pickle

import pandas as pd
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from transformers import GPT2Tokenizer

from .. import builder
from ..enums import Modality
from ..eval import evaluate_on_coco_caption, generate2
from ..models import Decoder
from ..parse_data import LABELS_JSONS_LST
from ..utils import (
    add_predictions_to_results_json,
    evaluate_list,
    get_metrics_out_filename,
    get_pred_filename,
)


class ClipCaptionLightningModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.decoder.model)

        self.model = Decoder(cfg)
        self.loss = builder.build_loss(cfg)
        self.prefix_length = self.model.prefix_length
        self.cfg = cfg

        self.input_modality = cfg.encoder.modality
        self.output_modality = cfg.decoder.modality

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.model)

        if not OmegaConf.is_none(self.cfg.train, "scheduler"):
            scheduler = builder.build_scheduler(self.cfg, optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        self.input_modality = Modality.Language
        loss, outputs = self.shared_loss_step(batch, split="train")
        return loss

    def validation_step(self, batch, batch_idx):
        if self.cfg.val_eval:
            self.input_modality = Modality.Vision
            out = self.quick_eval_step(batch, "val")
            self.validation_step_outputs.append(out)
        else:
            if self.cfg.cross_modal_val:
                self.input_modality = Modality.Vision
            else:
                self.input_modality = Modality.Language
            loss, outputs = self.shared_loss_step(batch, split="val")
            return loss

    def test_step(self, batch, batch_idx):
        self.input_modality = Modality.Vision
        out = self.eval_step(batch, split="test")
        self.test_step_outputs.append(out)

    def shared_loss_step(self, batch, split):
        prefix, labels, gold_caption, img_id, cap_id = batch

        if self.output_modality == Modality.Language:
            (labels, mask) = labels
            outputs = self.model(tokens=labels, prefix=prefix, mask=mask)
            outputs = outputs.logits[:, self.prefix_length - 1 : -1]
            loss = self.loss(outputs, labels)
        else:  # Vision
            outputs = self.model(x=prefix)
            loss = self.loss(outputs, labels, self.model)

        self.log(
            f"{split}_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss, outputs

    def eval_step(self, batch, split):
        # Note: batch size = 1
        prefix, labels, gold_caption, img_id, cap_id = batch

        prefix_embed = self.model.clip_project(prefix).reshape(
            -1, self.model.prefix_length, self.model.embed_size
        )
        pred = generate2(self.model, self.tokenizer, embed=prefix_embed)

        return {"image_id": img_id.item(), "caption": pred, "id": cap_id.item()}

    def quick_eval_step(self, batch, split):
        # Note: batch size = 1
        prefix, labels, gold_captions, img_id, cap_id = batch

        prefix_embed = self.model.clip_project(prefix).reshape(
            -1, self.model.prefix_length, self.model.embed_size
        )
        pred = generate2(self.model, self.tokenizer, embed=prefix_embed)

        return {"pred": pred, "refs": gold_captions}

    def on_validation_epoch_end(self):
        if self.cfg.val_eval and len(self.validation_step_outputs) > 0:
            preds = [o["pred"] for o in self.validation_step_outputs]
            refs = [o["refs"] for o in self.validation_step_outputs]

            # Save to files
            epoch = self.current_epoch if self.current_epoch else 0
            pred_file = get_pred_filename(self.cfg.output_dir, split="val", epoch=epoch)

            with open(pred_file, "wb") as f:
                pickle.dump({"preds": preds, "refs": refs}, f)

            print(f"=> Predictions at {pred_file}")

            scores = evaluate_list(preds, refs)

            for metric, val in scores.items():
                self.log(
                    f"val/{metric}", val, on_epoch=True, logger=True, prog_bar=True
                )

            if isinstance(self.logger, WandbLogger):
                for i, output in enumerate(self.validation_step_outputs[:10]):
                    pred = output["pred"]
                    refs = ["\n".join(r) for r in output["refs"]]

                    df = pd.DataFrame({"pred": pred, "refs": refs})
                    self.logger.log_text(key=f"generated_caption_{i}", dataframe=df)
            self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def shared_epoch_end(self, outputs, split):
        # Write predictions to json
        if split == "test":
            split = self.cfg.test_split

        epoch = self.current_epoch if self.current_epoch else 0

        # Compute eval metrics
        pred_file = get_pred_filename(self.cfg.output_dir, split, epoch=epoch)

        add_predictions_to_results_json(predictions=outputs, filepath=pred_file)

        print(f"=> Predictions at {pred_file}")

        out_file = get_metrics_out_filename(self.cfg.output_dir, split, epoch=epoch)
        if self.cfg.data.dataset == "coco":
            metrics_dict = evaluate_on_coco_caption(
                pred_file, LABELS_JSONS_LST[split], out_file
            )

        print(f"=> Metrics at {out_file}")

        # Log eval metrics
        for k, v in metrics_dict.items():
            self.log(f"{split}/{k}", v, on_epoch=True, logger=True, prog_bar=True)
