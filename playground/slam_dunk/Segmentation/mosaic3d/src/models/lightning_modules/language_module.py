import os
import random
import time
from typing import Any, Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torchmetrics import MaxMetric
from torchmetrics.classification.confusion_matrix import MulticlassConfusionMatrix

import src.utils.caption_utils as caption_utils
from src.models.lightning_modules.module_base import LitModuleBase
from src.models.losses.caption_loss import (
    CaptionAlignmentLoss,
    CaptionCLIPLoss,
    CaptionLoss,
    CaptionSigLIPLoss,
    DenseCaptionAlignmentLoss,
)
from src.models.losses.clip_alignment_loss import CLIPAlignmentEval
from src.models.utils.clip_models import build_clip_model, download_clip_model
from src.models.utils.evaluator import InstanceSegmentationEvaluator
from src.models.utils.structure import Point
from src.utils import RankedLogger

log = RankedLogger(__file__, rank_zero_only=True)


class DenseLanguageLitModule(LitModuleBase):
    def __init__(
        self,
        net,
        optimizer,
        scheduler,
        scheduler_interval: str,
        clip_encoder: Dict,
        compile: bool,
        loss_cfg: Dict,
        best_metric: str,
        eval_cfg: Optional[Dict] = None,
        use_prompt: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = None

        # Mix3D augmentations
        self.mix_prob = loss_cfg.get("mix_prob", 0)

        # loss functions
        self.caption_loss_type = loss_cfg["caption_loss"].get("type", "contrastive")
        if self.caption_loss_type == "contrastive":
            self.caption_loss = CaptionLoss(**loss_cfg["caption_loss"])
        elif self.caption_loss_type == "alignment":
            self.caption_loss = DenseCaptionAlignmentLoss(**loss_cfg["caption_loss"])
        elif self.caption_loss_type == "region_alignment":
            self.caption_loss = CaptionAlignmentLoss(**loss_cfg["caption_loss"])
        elif self.caption_loss_type == "clip":
            self.caption_loss = CaptionCLIPLoss(**loss_cfg["caption_loss"])
        elif self.caption_loss_type == "siglip":
            self.caption_loss = CaptionSigLIPLoss(**loss_cfg["caption_loss"])
        else:
            raise ValueError(f"Caption loss type {self.caption_loss_type} not supported")

        # for tracking best so far validation accuracy
        self.val_metrics = nn.ModuleDict()
        self.val_class_info = dict()
        self.val_dataset_names = dict()
        self.val_best_metric = MaxMetric()

        # Save val_best_metric to hparams and restore if resuming
        self.save_hyperparameters(ignore=["val_best_metric"])

        # Sync distributed metrics
        self.train_sync_dist = loss_cfg.get("sync_dist", False)

        # eval configs
        self.ignore_background = False
        self.ignore_class_prob = False

        # Inference-only options (GT metrics 없을 때)
        self.infer_only: bool = False
        self.infer_labels: Optional[List[str]] = None
        if eval_cfg is not None:
            self.infer_only = bool(eval_cfg.get("infer_only", False))
            self.infer_labels = eval_cfg.get("infer_labels", None)


    def prepare_data(self) -> None:
        # download clip model on rank 0
        ckpt_path = download_clip_model(self.hparams.clip_encoder)
        log.info(f"Downloaded CLIP model to {ckpt_path}")

    def configure_model(self) -> None:
        # network
        if self.net is not None:
            return

        self.net = self.hparams.net()
        # Print network on the first GPU
        if self.local_rank == 0:
            log.info(self.net)

        # clip encoder
        self.clip_encoder = build_clip_model(self.hparams.clip_encoder, device=self.device)

        # freeze clip encoder
        for params in self.clip_encoder.parameters():
            params.requires_grad = False

    def on_load_checkpoint(self, checkpoint):
        if hasattr(self.hparams, "val_best_metric"):
            value = checkpoint["hyper_parameters"].get("val_best_metric", None)
            if value is not None:
                self.val_best_metric.update(value.max_value)
        super().on_load_checkpoint(checkpoint)

    def setup(self, stage: str) -> None:
        val_dataloaders = self.trainer.datamodule.val_dataloader()
        if not isinstance(val_dataloaders, list):
            val_dataloaders = [val_dataloaders]

        for i, val_dataloader in enumerate(val_dataloaders):
            dataset = val_dataloader.dataset
            postfix = dataset.log_postfix
            assert postfix is not None, "log_postfix is required for clarity"

            # ---- class names 결정 (GT 없으면 infer_labels 사용) ----
            dataset_class_names = getattr(dataset, "CLASS_LABELS", None)
            if self.infer_only:
                class_names = self.infer_labels
                if not class_names or len(class_names) < 2:
                    raise ValueError(
                        "[infer_only] eval_cfg.infer_labels must be provided with >=2 labels. "
                        "Example: ['wall','floor','chair',...]"
                    )
            else:
                class_names = dataset_class_names
                if class_names is None or len(class_names) < 2:
                    raise ValueError(
                        f"[eval] dataset.CLASS_LABELS invalid (len={0 if class_names is None else len(class_names)}). "
                        "If this dataset has no GT semantic, run with model.eval_cfg.infer_only=true "
                        "and provide model.eval_cfg.infer_labels=[...]"
                    )

            # ---- metric 생성 (infer_only면 아예 안 만듦) ----
            val_metric = nn.ModuleDict()
            if not self.infer_only:
                val_metric["confmat"] = MulticlassConfusionMatrix(
                    num_classes=len(class_names),
                    ignore_index=dataset.ignore_label,
                )
                val_metric["confmat_all"] = MulticlassConfusionMatrix(
                    num_classes=len(class_names),
                    ignore_index=dataset.ignore_label,
                )

                # instance segmentation metrics
                if getattr(dataset, "mask_dir", None) is not None:
                    val_metric["mAP_evaluator"] = InstanceSegmentationEvaluator(
                        class_names=class_names,
                        segment_ignore_index=dataset.instance_ignore_class_idx
                        + [dataset.ignore_label],
                        instance_ignore_index=dataset.ignore_label,
                        subset_mapper=dataset.subset_mapper,
                    )

            # ---- class info (infer_only에서도 필요: clip text embedding용) ----
            val_class_info = dict(
                postfix=postfix,
                class_names=class_names,
                base_class_idx=getattr(dataset, "base_class_idx", None),
                novel_class_idx=getattr(dataset, "novel_class_idx", None),
                fg_class_idx=getattr(dataset, "fg_class_idx", list(range(len(class_names)))),
                bg_class_idx=getattr(dataset, "bg_class_idx", []),
                ignore_label=getattr(dataset, "ignore_label", -100),
                instance_ignore_class_idx=getattr(dataset, "instance_ignore_class_idx", None),
                subset_mapper=getattr(dataset, "subset_mapper", None),
            )

            self.val_metrics[postfix] = val_metric
            self.val_class_info[postfix] = val_class_info
            self.val_dataset_names[i] = postfix

        # clip alignment eval (labels가 있어야 하므로 infer_only에서도 생성)
        self.clip_alignment_eval = nn.ModuleDict(
            {
                postfix: CLIPAlignmentEval(**self.hparams.eval_cfg.seg_eval)
                for postfix in self.val_class_info.keys()
            }
        )


    def forward(self, batch: Any) -> Dict[str, Any]:
        point = self.net(batch)
        out_dict = self._output_to_dict(point, batch)
        return out_dict

    def _output_to_dict(self, output: Any, batch: Any) -> Dict[str, Any]:
        assert isinstance(output, Point)
        output: Point = output
        clip_feat = output.sparse_conv_feat.features[output.v2p_map]
        out_dict = dict(point=output, clip_feat=clip_feat)
        return out_dict

    def training_step(self, batch, batch_idx):
        self._train_start = time.time()

        if random.random() < self.mix_prob:
            offset = batch["offset"]
            batch["offset"] = torch.cat([offset[1:-1:2], offset[-1].unsqueeze(0)], dim=0)

        # Time forward pass
        self._forward_start = time.time()
        out_dict = self(batch)
        clip_feat = out_dict["clip_feat"]
        forward_time = time.time() - self._forward_start
        self.forward_time(forward_time)

        # loss
        caption_loss = 0

        # Time loss computation
        self._loss_start = time.time()

        caption_loss_kargs = {
            "captions": batch["caption_data"].get("caption", None),
            "embeddings": batch["caption_data"].get("embedding", None),
            "point_indices": batch["caption_data"]["point_indices"],
            "caption_offsets": batch["caption_data"]["caption_offsets"],
            "num_points_per_caption": batch["caption_data"]["num_points_per_caption"],
            "clip_encoder": self.clip_encoder,
        }
        caption_loss = (
            self.caption_loss.loss(clip_feat, **caption_loss_kargs)
            * self.hparams.loss_cfg.weights.caption_loss
        )

        loss = caption_loss
        loss_time = time.time() - self._loss_start
        self.loss_time(loss_time)

        lr = self.optimizers().param_groups[0]["lr"]
        log_metrics = dict(loss=loss, caption_loss=caption_loss, lr=lr)

        # useful metadata
        bs = len(batch["offset"]) - 1
        log_metrics["num_points"] = batch["coord"].shape[0] / bs
        log_metrics["num_objects"] = (batch["caption_data"]["caption_offsets"].shape[0] - 1) / bs

        # Calculate training time and mark start of next data loading
        train_time = time.time() - self._train_start
        self.train_time(train_time)
        self._data_load_start = time.time()

        # Add timing metrics to existing logging
        log_metrics.update(
            {
                "time/data_loading": self.data_load_time.compute(),
                "time/forward": self.forward_time.compute(),
                "time/loss": self.loss_time.compute(),
                "time/training": self.train_time.compute(),
            }
        )

        self.log_dict(
            {f"train/{key}": value for key, value in log_metrics.items()},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=self.train_sync_dist,
        )
        return loss

    def on_validation_epoch_start(self):
        # LitModuleBase가 val_results 등을 초기화할 가능성이 높으므로 호출
        try:
            super().on_validation_epoch_start()
        except Exception:
            pass

        # base가 안 만들었어도 여기서 강제로 생성
        self.val_results = []

        self.clip_encoder = self.clip_encoder.to(self.device)

        for postfix in self.val_class_info.keys():
            class_info = self.val_class_info[postfix]
            eval_module = self.clip_alignment_eval[postfix]
            class_names = class_info["class_names"]

            if eval_module.emb_target is None:
                if self.hparams.use_prompt:
                    class_names = [
                        f"a {c} in a scene" if "other" not in c else "other"
                        for c in class_names
                    ]
                text_embedding = caption_utils.forward_text_encoder(
                    class_names, self.clip_encoder, normalize=True, device=self.device
                )
                eval_module.set_target_embedding(text_embedding.to(self.device))
            else:
                if eval_module.emb_target.device != self.device:
                    eval_module.emb_target = eval_module.emb_target.to(self.device)

            metrics = self.val_metrics[postfix]
            for key in metrics.keys():
                metrics[key].reset()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        postfix = self.val_dataset_names[dataloader_idx]
        metrics = self.val_metrics[postfix]
        class_info = self.val_class_info[postfix]

        out_dict = self(batch)
        logits = self.clip_alignment_eval[postfix].predict(
            out_dict["clip_feat"], return_logit=True
        )

        # ----------------------------
        # Inference-only: GT 없어도 저장만 하고 종료
        # ----------------------------
        if self.infer_only or ("segment" not in batch):
            preds = logits.max(1)[1]

            if os.environ.get("SAVE_PRED", None) is not None:
                # ---- robust run dir resolution ----
                run_dir = None

                # 1) trainer.log_dir (Lightning provides this even with logger sometimes)
                if getattr(self, "trainer", None) is not None:
                    run_dir = getattr(self.trainer, "log_dir", None)

                    # 2) default_root_dir fallback
                    if not run_dir:
                        run_dir = getattr(self.trainer, "default_root_dir", None)

                # 3) hydra run dir fallback (usually cwd)
                if not run_dir:
                    run_dir = os.getcwd()

                pred_save_dir = os.path.join(run_dir, "pred")
                os.makedirs(pred_save_dir, exist_ok=True)

                torch.save(
                    {
                        "coord": batch["origin_coord"].cpu(),
                        "color": batch["color"].cpu() if "color" in batch else None,
                        "pc_count": batch["pc_count"].cpu(),
                        "pred": preds.cpu(),
                        "logits": logits.cpu(),
                        "class_names": class_info["class_names"],
                    },
                    os.path.join(pred_save_dir, f"pred_{batch_idx}.pth"),
                )

            return {"infer_only": True}


        # ----------------------------
        # 아래는 기존 정량평가용 코드
        # ----------------------------
        preds_all = logits.max(1)[1]
        metrics["confmat_all"](preds_all, batch["segment"])

        logits_fg = torch.full_like(logits, torch.finfo(logits.dtype).min)
        logits_fg[..., class_info["fg_class_idx"]] = logits[..., class_info["fg_class_idx"]]

        preds = logits_fg.max(1)[1]
        segment_fg = batch["segment"].clone()
        for i in class_info["bg_class_idx"]:
            segment_fg[segment_fg == i] = class_info["ignore_label"]

        metrics["confmat"](preds, segment_fg)

        if "mAP_evaluator" in metrics:
            self._update_instance_segmentation_metrics(batch, logits, metrics, class_info)

        return {"infer_only": False}

    def _update_instance_segmentation_metrics(self, batch, logits, metrics, class_info):
        offset = batch["offset"]
        batch_size = len(offset) - 1
        ignore_class_idx = class_info["instance_ignore_class_idx"]
        for i in range(batch_size):
            gt_classes = batch["segment"][offset[i] : offset[i + 1]]
            gt_instances = batch["instance"][offset[i] : offset[i + 1]]
            pred_logits = logits[offset[i] : offset[i + 1]]
            pred_masks = batch["masks_binary"][i]

            pred_logits_fg = pred_logits.clone()
            if self.ignore_background and ignore_class_idx is not None:
                pred_logits_fg[..., ignore_class_idx] = torch.finfo(pred_logits.dtype).min

            pred_logits_fg = torch.nn.functional.softmax(pred_logits_fg, dim=-1)
            pred_logits_fg = torch.stack([pred_logits_fg[mask].mean(dim=0) for mask in pred_masks])
            pred_scores, pred_classes = torch.max(pred_logits_fg, dim=1)

            if self.ignore_class_prob:
                pred_scores = torch.ones_like(pred_scores)

            metrics["mAP_evaluator"].update(
                pred_classes=pred_classes,
                pred_scores=pred_scores,
                pred_masks=pred_masks,
                gt_segment=gt_classes,
                gt_instance=gt_instances,
            )

    def on_validation_epoch_end(self) -> None:
        # infer_only면 metric 계산/로깅 스킵
        if self.infer_only:
            return
        super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)


    def children(self):
        for name, module in self.named_children():
            if name != "clip_encoder":
                yield module

    def parameters(self):
        for name, params in self.named_parameters():
            if "clip_encoder" not in name:
                yield params
