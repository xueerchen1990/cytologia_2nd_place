import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from metrics import compute_cytologia_metric_optimized
import argparse
import numpy as np
import os

def keep_top_k(arr, k, norm=True):
    """
    Processes a 2D array of probabilities (vectorized version).

    Args:
        arr: A NumPy array of shape (m, c) representing probabilities.
        k: The number of top probabilities to keep in each row.

    Returns:
        A NumPy array of the same shape, with top k probabilities kept 
        per row and rows normalized.
    """
    m, c = arr.shape

    # Find indices of top k elements in each row
    top_k_indices = np.argpartition(arr, -k, axis=1)[:, -k:]

    # Create a mask to keep only the top k elements
    mask = np.zeros_like(arr, dtype=bool)
    rows = np.arange(m)[:, np.newaxis]  # Row indices for broadcasting
    mask[rows, top_k_indices] = True

    # Keep only top k elements using the mask
    result = np.where(mask, arr, 0)

    # Normalize each row
    if norm:
        row_sums = result.sum(axis=1, keepdims=True)
        # Avoid division by zero; if a row sums to zero, it remains all zeros
        result = np.divide(result, row_sums, out=np.zeros_like(result), where=row_sums!=0)

    return result

def blend_predictions(valid_csv_path, prediction_csv_paths, weights=None, single_image_strategy='average'):
    """
    Blends predictions from multiple models for object detection and classification.

    Args:
        valid_csv_path (str): Path to the validation CSV file containing ground truth.
        prediction_csv_paths (list): List of paths to CSV files containing predictions from different models.
        weights (list, optional): List of weights for each model's predictions. If None, equal weights are used. Defaults to None.
        single_image_strategy (str): Strategy for handling images with only one bounding box in the ground truth. 
                                     Options: 'average', 'first', 'last' or the index of a model (int).
                                     Defaults to 'average' which averages the predictions of all models.
                                     'first' takes predictions from the first model in the list.
                                     'last' takes predictions from the last model in the list.
                                     An integer index selects the predictions from the model at that index in prediction_csv_paths.

    Returns:
        tuple: A tuple containing the blended DataFrame and the Cytology metric score.
    """

    warnings.filterwarnings("ignore")

    va = pd.read_csv(valid_csv_path)
    va['count'] = va.groupby('NAME')['NAME'].transform('count')
    mask = va['count'] == 1  # Identify images with only one bounding box
    

    # Load predictions and extract class probabilities and bounding box coordinates
    predictions = []
    for path in prediction_csv_paths:
        predictions.append(pd.read_csv(path))
    
    # m1 = predictions[0]['class'].isin(['BA', 'LAM3', 'Thromb', 'Lysee', 'Er', 'PNN', 'EO'])
    # mask = mask & (~m1)
    
    # m1 = predictions[0]['class'].isin(['BA', 'LAM3', 'Thromb', 'Lysee', 'Er', 'PNN', 'EO'])
    # mask = mask & (~m1)
    print('mask ratio', mask.sum()/va.shape[0])

    class_names = [i for i in predictions[0].columns[-23:]]
    # normalize class probabilities by divdiding by the sum of all class probabilities
    # for i in range(len(predictions)):
    #     #predictions[i][class_names] = keep_top_k(predictions[i][class_names].values, k=2)
    #     predictions[i][class_names] = predictions[i][class_names].div(predictions[i][class_names].sum(axis=1), axis=0)
    #box_cols = ['x1', 'y1', 'x2', 'y2']

    # Blend probabilities
    if weights is None:
        weights = [1.0] * len(predictions)
    else:
        assert len(weights) == len(predictions), "Number of weights must match the number of prediction files."

    total_weight = sum(weights)
    blended_probs = sum(
        predictions[i][class_names].values * weights[i] for i in range(len(predictions))
    ) / total_weight

    # Create blended DataFrame
    vb = predictions[0].drop(class_names, axis=1).copy()
    vb['class_id'] = vb['class'].map({i:c for c, i in enumerate(class_names)})
    # Handle single bounding box images based on the chosen strategy
    if single_image_strategy == 'average':
        blended_preds = blended_probs.argmax(axis=1)
        vb.loc[mask, 'class_id'] = blended_preds[mask]

    else:
        raise ValueError(f"Invalid single_image_strategy: {single_image_strategy}")

    vb['class'] = vb.class_id.map({c: i for c, i in enumerate(class_names)})
    vb['x1'] = vb.x1 + np.random.rand(vb.shape[0])*1e-6

    # Compute Cytology metric
    if 'val' in valid_csv_path:
        score, _ = compute_cytologia_metric_optimized(va, vb)
        #print(score)
    else:
        score = None
    return vb.drop('class_id', axis=1), score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Blend predictions from multiple models.")
    parser.add_argument("--data", type=str, choices=['val', 'test'], required=True,
                        help="Type of data to process: 'val' or 'test'")
    parser.add_argument("--tag", type=str, required=True,
                        help="model tag(s), separated by comma if multiple")
    parser.add_argument("--folds", type=int, required=False, default=4,
                        help="number of folds")
    args = parser.parse_args()
    models = args.tag.split(',')
    
    # Create blend directory if it doesn't exist
    os.makedirs("./blend", exist_ok=True)

    if args.data == 'val':
        va = []
        for i in range(args.folds):
            va.append(pd.read_csv(f'{PATH}/det_cv/fold_{i}/val.csv'))
        va = pd.concat(va)
        valid_csv_path = './blend/val.csv'
        va.to_csv(valid_csv_path, index=False)
        
        
        # Iterate through each model and its args.folds folds, concatenating and summing probabilities
        outs = []
        for model in models:
            pdf = []
            for i in range(args.folds):
                fold_df = pd.read_csv(f'./save/{model}/fold_{i}/val_res.csv')
                pdf.append(fold_df)
            pdf = pd.concat(pdf)
            out = f'./blend/val_{model}.csv'
            pdf.to_csv(out, index=False)
            outs.append(out)
        df, score = blend_predictions(valid_csv_path, outs)
        print(score)
    elif args.data == 'test':
        valid_csv_path = f'{PATH}/test.csv'
        predictions = [f'./save/{model}/fold_{i}/test_res.csv' for i in range(4) for model in models]
        blended_df,_ = blend_predictions(valid_csv_path, predictions)
        output_filename = f"./blend/test_{args.tag}_avg.csv".replace(',','_')
        blended_df.to_csv(output_filename, index=False)
        print(f"Saved blended test results to {output_filename}")