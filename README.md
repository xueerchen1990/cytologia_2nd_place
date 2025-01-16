# CytologIA 2nd Place Solution

This repository contains the 2nd place solution for the [CytologIA Data Challenge: Advancing Hematological Diagnostics Through AI](https://www.trustii.io/post/announcing-the-cytologia-data-challenge-advancing-hematological-diagnostics-through-ai?ref=mlcontests).

## Overview

This solution utilizes YOLO models for both detection and classification of cytological images and ensemble their predictions. The code is structured to train models across multiple folds for cross-validation and includes scripts for data preparation, model training, and inference.

## Repository Structure

-   **`create_cv.py`**: Script for creating cross-validation folds and preparing data for YOLO models.
-   **`train_det.py`**: Script for training YOLO detection models.
-   **`train_cls_crop.py`**: Script for training YOLO classification models, using crops of cells.
-   **`train_cls_rotate.py`**: Script for training YOLO classification models, using rotated versions of cells.
-   **`focal_trainer.py`**: Contains custom trainer class that utilizes focal loss for classification, subclassed from Ultralytics' ClassificationTrainer.
-   **`focal_trainer_crop.py`**: Defines a custom trainer class based on `ClassificationTrainerFocalLoss`, optimized for cropped cell images.
-   **`focal_trainer_rotate.py`**: Defines another custom trainer class, `ClassificationTrainerFocalLoss`, tailored for handling rotated cell images.
-   **`utils.py`**: Utility functions including environment variable loading, YAML parsing, and image flipping.
-   **`train.sh`**: Shell script to train the models.
-   **`inference.sh`**: Shell script to perform inference using the trained models.

## Data Preparation

```python
python create_cv.py
```

## Model Training

### Detection Model

The `train_det.py` script is used to train the detection model using the `CustomDetectionTrainer`. Key functions and features include:

-   `generalized_box_iou()`: Calculates the Generalized Intersection over Union (IoU) for bounding boxes.
-   `train()`: Trains a YOLO detection model with specified parameters.

To train the detection model, modify `train.sh` to call `train_det.py` with the appropriate arguments.

### Classification Model

#### Crop-based Classification

The `train_cls_crop.py` script trains a classification model using crops of cell images. It uses the `ClassificationTrainerFocalLoss` from `focal_trainer_crop.py`, which is a custom trainer with focal loss.

#### Rotation-based Classification

The `train_cls_rotate.py` script trains a classification model using rotated versions of cell images. It also uses the `ClassificationTrainerFocalLoss` from `focal_trainer_rotate.py`.

### Training Script

The `train.sh` script is the entry point for training:

```bash
# To train
./train.sh
```

## Inference
Please refer to the [instructions for local inference](https://github.com/xueerchen1990/cytologia_2nd_place/blob/main/notebooks/README.md) to reproduce the best submission

## Utility Functions

-   `utils.py`:
    -   `load_env()`: Loads environment variables from a `.env` file.
    -   `get_yaml_value()`: Retrieves a value from a YAML file based on a given key.
    -   `flip4()`: Generates four flipped versions of an image (original, left-right, top-bottom, both).

## Main Dependencies

-   Python 3.10+
-   Ultralytics YOLO
-   pandas
-   NumPy
-   tqdm

## Notes

-   The code assumes a specific directory structure for the dataset, with paths like `/raid/ml/cytologia/`.
-   You can change data path in constants.py
