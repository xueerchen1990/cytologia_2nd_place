# in ultralytics/utils/metrics.py
import os
import argparse

parser = argparse.ArgumentParser(description="Train yolo detection model.")
# parser.add_argument("--yaml", type=str, required=True,
#                     help="yaml file path")
parser.add_argument('--model', type=str, help='pretrained model')
parser.add_argument('--tag', type=str, help='tag of config')
parser.add_argument('--gpu', type=str, default='0', help='gpu id to run inference')
parser.add_argument('--fold', type=int, help='fold id')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
from ultralytics.utils import LOGGER
from sklearn.metrics import f1_score
from ultralytics import YOLO
from constants import PATH

def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Generalized IoU between two sets of bounding boxes.

    Args:
        boxes1: Tensor of shape (N, 4) with boxes in [x1, y1, x2, y2] format.
        boxes2: Tensor of shape (M, 4) with boxes in [x1, y1, x2, y2] format.

    Returns:
        A tensor of shape (N, M) with GIoU values.
    """
    # Intersection area
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Areas of each box
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Union area
    union_area = area1[:, None] + area2 - inter_area

    # IoU
    iou = inter_area / union_area.clamp(min=1e-7)

    # Enclosure (smallest enclosing box)
    enc_x1 = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    enc_y1 = torch.min(boxes1[:, None, 1], boxes2[:, 1])
    enc_x2 = torch.max(boxes1[:, None, 2], boxes2[:, 2])
    enc_y2 = torch.max(boxes1[:, None, 3], boxes2[:, 3])

    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    # GIoU
    giou = iou - (enc_area - union_area) / enc_area.clamp(min=1e-7)
    return giou

class CytologiaMetric:

    def update(self, pred_bboxes, pred_classes, pred_scores, gt_bboxes, gt_classes, image_name):
        """
        Update the metric with data from a single image. Cache predictions and ground truths for later F1 computation.
        """
        num_gt = gt_bboxes.shape[0]
        num_pred = pred_bboxes.shape[0]

        if num_gt == 0 or num_pred == 0:
            if num_gt > 0:
                # If there are ground truths but no predictions, add placeholders for predictions to keep lengths consistent
                self.pred_classes_cache.extend([-1] * num_gt) # Use -1 or any other invalid class index to indicate no prediction
                self.gt_classes_cache.extend(gt_classes.cpu().numpy())
                self.all_giou_sum += -num_gt
                self.image_count += num_gt
            return

        # Compute GIoU
        giou = generalized_box_iou(gt_bboxes, pred_bboxes[:len(gt_bboxes)])

        # Find best prediction for each ground truth
        best_pred_indices = giou.argmax(dim=1)
        best_giou_values = giou.max(dim=1).values

        # Calculate rescaled GIoU
        rescaled_giou = (best_giou_values + 1) / 2
        self.all_giou_sum += rescaled_giou.sum().item()
        self.image_count += num_gt

        # Cache predicted and ground truth classes for F1 calculation
        best_pred_classes = pred_classes[best_pred_indices]
        self.pred_classes_cache.extend(best_pred_classes.cpu().numpy())
        self.gt_classes_cache.extend(gt_classes.cpu().numpy())

    def compute_final_metric(self):
        """Compute the final Cytologia metric, including macro F1 using sklearn."""
        if self.image_count > 0:
            mean_giou = self.all_giou_sum / self.image_count

            # Compute macro F1 using sklearn
            # Ensure that there are no -1 values in pred_classes_cache if using f1_score
            # Replace -1 with a valid class index or filter out such cases
            temp_pred_classes = [p if p != -1 else 0 for p in self.pred_classes_cache] # Example: replace -1 with 0

            if -1 in self.pred_classes_cache:
                LOGGER.warning("Warning: Placeholder class index -1 was replaced with 0 when calculating macro F1.")

            macro_f1 = f1_score(self.gt_classes_cache, temp_pred_classes, average='macro')

            self.results_dict["cytology"] = 0.2 * mean_giou + 0.8 * macro_f1
            self.results_dict["giou"] = mean_giou
            self.results_dict["f1"] = macro_f1

    def reset(self):
        self.results_dict = {"cytology": 0.0, "giou": 0, "f1": 0}
        self.all_giou_sum = 0
        self.image_count = 0
        self.pred_classes_cache = []
        self.gt_classes_cache = []

    def mean_results(self):
        return [self.results_dict[i] for i in self.keys]

    @property
    def keys(self):
        """Return keys for Cytologia metrics."""
        return ["cytology", "giou", "f1"]

import torch
import numpy as np
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER
from pathlib import Path

class CustomDetectionValidator(DetectionValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize CustomDetectionValidator, inheriting from DetectionValidator."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.metrics = CytologiaMetric()

    def init_metrics(self, model):
        """Initialize CytologiaMetric instead of DetMetrics."""
        super().init_metrics(model)
        self.metrics.reset()

    def update_metrics(self, preds, batch):
        """Updates metrics statistics and results dictionary with predictions and ground truth from a batch."""
        for si, pred in enumerate(preds):
            self.seen += 1
            num_pred = len(pred)
            target_data = self._prepare_batch(si, batch)
            gt_bboxes, gt_classes = target_data.pop("bbox"), target_data.pop("cls")
            nl = len(gt_classes)

            if num_pred == 0:
                print("empty", si)
                if nl:
                    # If no predictions and there are labels, pass empty tensors to metrics
                    self.metrics.update(
                        torch.tensor([], device=self.device),  # Empty pred_bboxes
                        torch.tensor([], device=self.device),  # Empty pred_classes
                        torch.tensor([], device=self.device),  # Empty pred_scores
                        gt_bboxes,
                        gt_classes,
                        Path(batch["im_file"][si]).name
                    )
                continue

            # Prepare predictions
            predn = self._prepare_pred(pred, target_data)
            pred_bboxes, pred_classes, pred_scores = predn[:, :4], predn[:, 5], predn[:, 4]

            # Update custom metrics
            self.metrics.update(
                pred_bboxes,
                pred_classes,
                pred_scores,
                gt_bboxes,
                gt_classes,
                Path(batch["im_file"][si]).name
            )

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        self.metrics.compute_final_metric()
        res = {k:v for k,v in self.metrics.results_dict.items()}
        res['fitness'] = res["cytology"]
        return res

    def print_results(self):
        """Prints training/validation set metrics per class."""
        pf = "%22s" + "%11i" + "%11.3g" * (len(self.metrics.keys))  # print format
        LOGGER.info(pf % ("all", self.seen,  *self.metrics.mean_results()))

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 4) % ("Class", "Images", "cytology", "giou", "f1")

from ultralytics.models.yolo.detect import DetectionTrainer
from copy import copy

class CustomDetectionTrainer(DetectionTrainer):
    def get_validator(self):
        """Returns a DetectionValidator instance for model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return CustomDetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

def train(model_name, tag, fold, save_dir):
    model = YOLO(model_name)

    model.train(
        trainer=CustomDetectionTrainer,
        data=os.path.join(f"{PATH}/det_cv", f'fold_{fold}', 'data.yaml'),
        save_dir=os.path.join(save_dir, tag, model_name, f'fold_{fold}'),
        epochs=100,
        imgsz=360,
        augment=True,
        hsv_h = 0,
        hsv_s = 0,
        hsv_v = 0,
        flipud = 0.5,
        max_det=11,
        project=save_dir,  # wandb project name
        name=tag+'/'+f'fold_{fold}',      # wandb run name
        )

if __name__ == '__main__':
    train(args.model, args.tag, args.fold, 'save')