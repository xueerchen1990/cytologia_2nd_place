import torch.nn as nn
import torch
import torch.nn.functional as F
from copy import copy
import torch
from sklearn.metrics import f1_score
from ultralytics.utils.metrics import ClassifyMetrics
from ultralytics.models.yolo.classify.val import ClassificationValidator
from ultralytics.utils import LOGGER
import cv2
import pandas as pd
from constants import PATH

class CustomClassifyMetrics(ClassifyMetrics):
    """
    Custom classification metrics class to add F1 score as a metric and use it as fitness.
    """

    def __init__(self) -> None:
        """Initialize CustomClassifyMetrics instance."""
        super().__init__()
        self.f1 = 0

    def process(self, targets, pred):
        """
        Processes the targets and predictions to compute classification metrics including F1 score.

        Args:
            targets (torch.Tensor): Ground truth labels.
            pred (torch.Tensor): Predicted labels.
        """
        super().process(targets, pred)
        # Compute F1 score using sklearn
        # Convert tensors to numpy arrays
        targets_np = torch.cat(targets).numpy()
        pred_np = torch.cat(pred)[:,0].numpy()  # Assuming pred is one-hot encoded, get argmax for labels

        self.f1 = f1_score(targets_np, pred_np, average="macro")

    @property
    def fitness(self):
        """Returns F1 score as fitness score."""
        return self.f1

    @property
    def results_dict(self):
        """Returns a dictionary with model's performance metrics and fitness score, including F1 score."""
        return dict(
            zip(
                self.keys + ["fitness", "metrics/f1"],
                [self.top1, self.top5, self.fitness, self.f1],
            )
        )

    @property
    def keys(self):
        """Returns a list of keys for the results_dict property, including F1 score."""
        return super().keys + ["metrics/f1"]

class CustomClassificationValidator(ClassificationValidator):
    """
    Custom classification validator to use the custom metrics with F1 score.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initializes CustomClassificationValidator instance."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.metrics = CustomClassifyMetrics()
    
    def get_desc(self):
        """Returns a formatted string summarizing classification metrics including F1 score."""
        return ("%22s" + "%11s" * 3) % ("classes", "top1_acc", "top5_acc", "f1")

    def print_results(self):
        """Prints evaluation metrics for YOLO object detection model, including F1 score."""
        pf = "%22s" + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.metrics.top1, self.metrics.top5, self.metrics.f1))

class ClassificationFocalLoss(nn.Module):
    """Focal loss for classification tasks."""

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        """
        Initializes the FocalLoss.

        Args:
            gamma (float): Focusing parameter. gamma > 0 reduces the relative loss
                          for well-classified examples, putting more focus on hard,
                          misclassified examples.
            alpha (float or list): Weighting factor in range (0,1) to balance
                                   positive vs negative examples or a list of
                                   weights for each class. If None, no weighting is applied.
            reduction (str): The reduction to apply to the output: 'none' | 'mean' | 'sum'.
                             'none': no reduction will be applied,
                             'mean': the weighted mean of the output is taken,
                             'sum': the output will be summed.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute the focal loss.

        Args:
            inputs (torch.Tensor): Predicted class probabilities or logits.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: The computed focal loss.
        """
        # Ensure targets are in one-hot representation
        preds = inputs
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        targets =  targets['cls']
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=preds.shape[1]).float()

        # Compute the cross-entropy loss
        ce_loss = F.cross_entropy(preds, targets, reduction="none")

        # Compute the focal loss
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (int, float)):
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            elif isinstance(self.alpha, list):
                alpha_t = torch.tensor(self.alpha, device=targets.device).gather(0, targets.argmax(1))
                alpha_t = alpha_t.view(-1, 1).expand_as(focal_loss)
            else:
                raise ValueError("alpha must be either a float/int or a list")

            focal_loss = alpha_t * focal_loss

        # Apply the specified reduction
        if self.reduction == "mean":
            loss = focal_loss.mean()
        elif self.reduction == "sum":
            loss = focal_loss.sum()
        else:  # none
            loss = focal_loss
        return loss, loss.detach()

# Add to ultralytics/nn/tasks.py
# (Make sure to also include necessary imports at the top of tasks.py, e.g., from ultralytics.utils.loss import ClassificationFocalLoss)
from ultralytics.nn.tasks import ClassificationModel
class ClassificationModelFocalLoss(ClassificationModel):
    """YOLOv8 classification model with focal loss."""

    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return ClassificationFocalLoss() # (add cls_alpha and cls_gamma to your config if you want to tune)

# Add to ultralytics/models/yolo/classify/train.py
# (Make sure to also include necessary imports at the top of train.py, e.g., from ultralytics.nn.tasks import ClassificationModelFocalLoss)

from PIL import Image
import random

def random_crop_with_bbox(image, bbox, scale=1.0):
    """
    Randomly crops an image while ensuring the entire bbox remains within the cropped region.
    Maintains the aspect ratio of the original image.
    
    Args:
    image: A PIL Image object.
    bbox: A tuple representing the bounding box (x1, y1, x2, y2).
    scale: A float between 0 and 1 representing the desired scaling factor.
           The actual scale will be the maximum of this value and the bbox's
           area relative to the image's area, ensuring the bbox is contained.
    
    Returns:
    A PIL Image object representing the cropped image.
    """
    try:
        img_width, img_height = image.size
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Calculate the minimum scale required to contain the entire bbox
        min_scale = max(bbox_width / img_width, bbox_height / img_height)
        
        # Use the larger of the provided scale and the minimum scale
        actual_scale = max(scale, min_scale)
        
        # Calculate the dimensions of the cropped region
        crop_width = int(img_width * actual_scale)
        crop_height = int(img_height * actual_scale)
        
        # Determine the valid range for the top-left corner of the crop
        max_x = img_width - crop_width
        max_y = img_height - crop_height
        
        # Ensure the bbox is contained within the crop's range
        min_crop_x = min(max(0, x2 - crop_width), max_x)  
        max_crop_x = max(0, min(x1, max_x))
        
        min_crop_y = min(max(0, y2 - crop_height), max_y)
        max_crop_y = max(0, min(y1, max_y))
        
        # Randomly select the top-left corner of the crop
        crop_x = random.randint(min_crop_x, max_crop_x)
        crop_y = random.randint(min_crop_y, max_crop_y)
        
        # Perform the crop
        cropped_image = image.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
    except:
        return image
    return cropped_image

from pathlib import Path

from ultralytics.data import ClassificationDataset
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.utils import LOGGER
import numpy as np
class BBoxClassificationDataset(ClassificationDataset):
    """
    A custom dataset class that extends ClassificationDataset to include bounding box center cropping augmentation.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the BBoxClassificationDataset with optional configuration for bounding box augmentation.
        """
        self.use_bbox_crop = kwargs.pop("use_bbox_crop", False)
        super().__init__(*args, **kwargs)
        fold = kwargs['args'].data.split('_')[-1]
        train = pd.read_csv(f'{PATH}/train.csv')
        va = pd.read_csv(f'{PATH}/det_cv/fold_{fold}/val.csv')
        mask = train.NAME.isin(va.NAME.unique())
        tr = train[~mask]
        self.name_coord_dict = {}
        for index, row in tr.iterrows():
            self.name_coord_dict[row['NAME']] = (row['x1'], row['y1'], row['x2'], row['y2'])
        

    def __getitem__(self, i):
        """
        Returns a single data sample from the dataset, potentially augmented with bounding box center cropping.
        """
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram:
            if im is None:  # Warning: two separate if statements required here, do not combine this with previous line
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR

        if self.use_bbox_crop:
            # Generate a random bounding box for augmentation
            height, width = im.shape[:2]
            img_name = f.split('/')[-1]
            bbox = self.name_coord_dict[img_name]

            # Apply draw_bbox_crop augmentation
            scale = 0.7+random.random()/4
            im = random_crop_with_bbox(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), bbox, scale)
            # plt.imshow(im)
            if im is None:
                LOGGER.warning(
                    "WARNING ⚠️ Bounding box augmentation resulted in an invalid crop. "
                    "Using original image instead."
                )
                im = Image.fromarray(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB))
        else:
            # Convert NumPy array to PIL image
            im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

from ultralytics.models.yolo.classify import ClassificationTrainer
from ultralytics.utils import DEFAULT_CFG,RANK
class ClassificationTrainerFocalLoss(ClassificationTrainer):
    """
    A class extending the ClassificationTrainer class for training classification models with focal loss.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a ClassificationTrainerFocalLoss object with optional configuration overrides and callbacks."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "classify"
        if overrides.get("imgsz") is None:
            overrides["imgsz"] = 224
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Returns a modified PyTorch model configured for training YOLO with focal loss."""
        model = ClassificationModelFocalLoss(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        # Copy hyperparameter args from ClassificationTrainer to ClassificationModelFocalLoss
        if hasattr(self, "args"):
            model.args = self.args
        
        for m in model.modules():
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout  # set dropout
        for p in model.parameters():
            p.requires_grad = True  # for training
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Creates a BBoxClassificationDataset instance given an image path, and mode (train/test etc.).
        """
        return BBoxClassificationDataset(
            root=img_path,
            args=self.args,
            augment=False,
            prefix=mode,
            use_bbox_crop=mode == "train"
        )

    def get_validator(self):
        """Returns a DetectionValidator instance for model validation."""
        self.loss_names = ["loss"]
        return CustomClassificationValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)
