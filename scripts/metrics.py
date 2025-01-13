import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from utils import timer,get_all_img_sizes
from constants import PATH

def compute_cytologia_metric(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    images_df = pd.DataFrame()

    for image in true_df["NAME"].unique():
        true_mask = true_df["NAME"] == image
        true_image_df = true_df[true_mask]

        pred_mask = (pred_df["NAME"] == image) & \
                    (pred_df["trustii_id"].isin(true_image_df["trustii_id"].unique()))
        pred_image_df = (
            pred_df[pred_mask]
            .drop_duplicates(subset="trustii_id")
            .drop_duplicates(subset=["x1", "y1", "x2", "y2", "class"])
        )

        if pred_image_df.shape[0] > 0:
            giou = generalized_box_iou(
                true_image_df[["x1", "y1", "x2", "y2"]].values,
                pred_image_df[["x1", "y1", "x2", "y2"]].values,
            )

            result_indices, result_values = find_max_index_values_by_row(giou)

            pred_image_df["boundingbox_id"] = range(len(pred_image_df))
            true_image_df["boundingbox_id"] = result_indices
            true_image_df["giou"] = result_values

            image_df = true_image_df.merge(
                pred_image_df[["boundingbox_id", "class"]], 
                how="left", 
                on="boundingbox_id"
            )
        else:
            image_df = true_image_df.copy().rename(columns={"class": "class_x"})
            image_df["boundingbox_id"] = range(len(image_df))
            image_df["class_y"] = None
            image_df["giou"] = -1

        images_df = pd.concat([images_df, image_df])

    images_df["rescaled_giou"] = ((images_df["giou"] + 1) / 2).fillna(0)
    f_score = f1_score(
        images_df["class_x"], images_df["class_y"].astype(str), average="macro"
    )
    print(images_df["rescaled_giou"].mean(), f_score)
    return float((0.2 * images_df["rescaled_giou"].mean() + 0.8 * f_score)), images_df

def generalized_box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calculate the Generalized IoU between two sets of bounding boxes.

    Args:
        boxes1: Numpy array of shape (N, 4) with boxes in [x1, y1, x2, y2] format.
        boxes2: Numpy array of shape (M, 4) with boxes in [x1, y1, x2, y2] format.

    Returns:
        A numpy array of shape (N, M) with GIoU values.
    """
    # Intersection area
    inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])

    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)

    # Areas of each box
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Union area
    union_area = area1[:, None] + area2 - inter_area

    # IoU
    iou = inter_area / np.maximum(union_area, 1e-7)

    # Enclosure (smallest enclosing box)
    enc_x1 = np.minimum(boxes1[:, None, 0], boxes2[:, 0])
    enc_y1 = np.minimum(boxes1[:, None, 1], boxes2[:, 1])
    enc_x2 = np.maximum(boxes1[:, None, 2], boxes2[:, 2])
    enc_y2 = np.maximum(boxes1[:, None, 3], boxes2[:, 3])

    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    # GIoU
    giou = iou - (enc_area - union_area) / np.maximum(enc_area, 1e-7)
    return giou

def find_max_index_values_by_row(matrix: np.ndarray) -> tuple:
    """
    Find the index and value of the maximum element in each row of a 2D array.

    Args:
        matrix: A numpy 2D array.

    Returns:
        A tuple containing:
            - An array of indices of the max values for each row.
            - An array of the max values for each row.
    """
    max_indices = np.argmax(matrix, axis=1)
    max_values = np.max(matrix, axis=1)
    return max_indices, max_values

def generalized_box_iou_vectorized(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calculate the Generalized IoU between two sets of bounding boxes element-wise.

    Args:
        boxes1: Numpy array of shape (N, 4) with boxes in [x1, y1, x2, y2] format.
        boxes2: Numpy array of shape (N, 4) with boxes in [x1, y1, x2, y2] format.

    Returns:
        A numpy array of shape (N,) with GIoU values.
    """
    # Ensure that both arrays have the same shape
    if boxes1.shape != boxes2.shape:
        raise ValueError("Input arrays must have the same shape.")
    
    # Intersection area
    inter_x1 = np.maximum(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = np.maximum(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = np.minimum(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = np.minimum(boxes1[:, 3], boxes2[:, 3])

    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)

    # Areas of each box
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Union area
    union_area = area1 + area2 - inter_area

    # IoU
    iou = inter_area / np.maximum(union_area, 1e-7)

    # Enclosure (smallest enclosing box)
    enc_x1 = np.minimum(boxes1[:, 0], boxes2[:, 0])
    enc_y1 = np.minimum(boxes1[:, 1], boxes2[:, 1])
    enc_x2 = np.maximum(boxes1[:, 2], boxes2[:, 2])
    enc_y2 = np.maximum(boxes1[:, 3], boxes2[:, 3])

    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    # GIoU
    giou = iou - (enc_area - union_area) / np.maximum(enc_area, 1e-7)
    return giou


@timer
def compute_cytologia_metric_optimized(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    # 1. Identify images with a single bounding box
    bbox_counts = true_df.groupby("NAME")["trustii_id"].count()
    single_bbox_images = bbox_counts[bbox_counts == 1].index

    # 2. Create a mask for single-bbox images
    single_bbox_mask = true_df["NAME"].isin(single_bbox_images)

    # 3. Handle single-bbox images with vectorized operations
    true_single_df = true_df[single_bbox_mask].copy()
    pred_single_df = pred_df[single_bbox_mask].copy()

    if not true_single_df.empty:
        true_boxes = true_single_df[["x1", "y1", "x2", "y2"]].values
        pred_boxes = pred_single_df[["x1", "y1", "x2", "y2"]].values

        # Vectorized GIoU calculation
        giou_values = generalized_box_iou_vectorized(true_boxes, pred_boxes)

        true_single_df["giou"] = giou_values
        true_single_df["class_y"] = pred_single_df["class"].values
        true_single_df["boundingbox_id"] = 0
        # rename class to class_x
        true_single_df.rename(columns={"class": "class_x"}, inplace=True)

    # 4. Handle multi-bbox images with the ORIGINAL logic
    images_df_list = []
    if (~single_bbox_mask).any():  # Check if there are any multi-bbox images
        _, images_df_multi = compute_cytologia_metric(
            true_df[~single_bbox_mask], pred_df[~single_bbox_mask]
        )
        images_df_list.append(images_df_multi)
    
    # 5. Combine the results
    images_df_list.append(true_single_df)  # Add single-bbox results at the beginning
    images_df = pd.concat(images_df_list)

    # 6. Calculate the final metric (same as before)
    images_df["rescaled_giou"] = ((images_df["giou"] + 1) / 2).fillna(0)
    f_score = f1_score(
        images_df["class_x"], images_df["class_y"].astype(str), average="macro"
    )
    print(images_df["rescaled_giou"].mean(), f_score)
    return float((0.2 * images_df["rescaled_giou"].mean() + 0.8 * f_score)), images_df

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    import os
    import argparse
    parser = argparse.ArgumentParser(description="evaluation")

    # Add arguments to the parser
    parser.add_argument('--tag', type=str, help='tag of config')
    parser.add_argument('--save_dir', type=str, help='dir where weights are saved', default='./save')
    parser.add_argument('--data_dir', type=str, help='dir where images live', default=f'{PATH}/det_cv')
    parser.add_argument('--fold', type=int, help='fold id')
    # Parse the command-line arguments
    args = parser.parse_args()

    gt_path = os.path.join(args.data_dir, f"fold_{args.fold}/val.csv")
    pred_path = os.path.join(args.save_dir, f"{args.tag}/fold_{args.fold}/val_res.csv")

    va = pd.read_csv(gt_path)
    vb = pd.read_csv(pred_path)
    va = get_all_img_sizes(va, '/'.join(args.data_dir.split('/')[:-1])+'/images')
    va['count'] = va.groupby('NAME')['NAME'].transform('count')


    mask = va['count'] == 1
    score,_ = compute_cytologia_metric_optimized(va[mask], vb[mask])
    print('bbox_count=1')
    print(score)
    print()

    rm = ['BA', 'LAM3', 'Thromb', 'Lysee', 'Er', 'PNN', 'EO']
    m2 = vb['class'].isin(rm) | va['class'].isin(rm)
    mask = mask & (~m2)
    score,_ = compute_cytologia_metric_optimized(va[mask], vb[mask])
    print('hard classes & bbox_count=1')
    print(score)
    print()


    # if 'cls' in args.tag:
    #     va['bbox_count'] = va.groupby('NAME')['NAME'].transform('count')
    #     mask = va.bbox_count == 1
    #     va = va[mask]
    #     vb = vb[mask]
    score,_ = compute_cytologia_metric_optimized(va, vb)
    print('all')
    print(score)
