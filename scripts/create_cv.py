from sklearn.model_selection import StratifiedGroupKFold

import os
import pandas as pd
import numpy as np  
from PIL import Image
from tqdm import tqdm
import yaml
from utils import draw_bounding_box, get_all_img_sizes
from constants import PATH

def get_class_id(dataframe):
    class_to_id = sorted(dataframe['class'].unique().tolist())
    class_to_id = {cls: idx for idx, cls in enumerate(class_to_id)}
    return class_to_id

def write_yolo_labels(dataframe, images_dir, labels_output_dir):
    class_to_id = get_class_id(dataframe)
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        image_name = row['NAME']
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        class_name = row['class']
        
        # Source image path
        image_path = os.path.join(images_dir, image_name)
        
        # Check if the image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_name} not found.")
            assert 0
        
        # Open the image to get its size
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error opening image {image_name}: {e}.")
            assert 0
        
        # Convert bounding box to YOLO format
        box_width = x2 - x1
        box_height = y2 - y1
        center_x = x1 + box_width / 2
        center_y = y1 + box_height / 2

        def fix(x):
            x = min(x, 1)
            x = max(x, 0)
            return x

        # Normalize the coordinates by the image size
        center_x_norm = fix(center_x / img_width)
        center_y_norm = fix(center_y / img_height)
        width_norm = fix(box_width / img_width)
        height_norm = fix(box_height / img_height)
        
        # Get the class ID
        class_id = class_to_id.get(class_name)
        if class_id is None:
            print(f"Warning: Class {class_name} not found in class mapping. Skipping.")
            continue
        
        # Prepare the label line
        label_line = f"{class_id} {center_x_norm:.6f} {center_y_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n"
        
        # make output dir
        os.makedirs(labels_output_dir, exist_ok=True)
        label_filename = os.path.splitext(image_name)[0] + '.txt'
        label_output_path = os.path.join(labels_output_dir, label_filename)
        
        # Append the label to the file
        with open(label_output_path, 'a') as label_file:
            label_file.write(label_line)


def create_cv(train, images_dir, labels_dir, output_dir):
    sgkf = StratifiedGroupKFold(n_splits=4)
    class_to_id = get_class_id(train)
    for i, (train_index, test_index) in enumerate(sgkf.split(train, train['class'], train['NAME'])):
        tr,va = train.loc[train_index], train.loc[test_index]
        tr = tr.reset_index(drop=True)
        va = va.reset_index(drop=True)
        mask = va.NAME.isin(tr.NAME.unique())
        print('Fold:', i, tr.shape, va.shape, tr.NAME.unique().shape)
        print(mask.sum())
        diff = (tr['class'].value_counts()/tr.shape[0]).values - (va['class'].value_counts()/va.shape[0]).values
        print(diff.max())

        for tag in ['images','labels']:
            for sub_tag in ['train','val']:
                out_dir = os.path.join(output_dir, f'fold_{i}', tag, sub_tag)
                os.makedirs(out_dir, exist_ok=True)

        def copy_files(df, tag):
            imgs = df.NAME.unique()
            for img in tqdm(imgs):
                src = os.path.join(images_dir, img)
                dst = os.path.join(output_dir, f'fold_{i}', 'images', tag, img)
                cmd = f'ln -s {src} {dst}'
                os.system(cmd)
                
                label = img.replace('.jpg', '.txt')
                src = os.path.join(labels_dir, label)
                dst = os.path.join(output_dir, f'fold_{i}', 'labels', tag, label)
                cmd = f'ln -s {src} {dst}'
                os.system(cmd)
        copy_files(tr, 'train')
        copy_files(va, 'val')
        data_yaml = {
            'train': os.path.join(output_dir, f'fold_{i}', 'images', 'train'),
            'val': os.path.join(output_dir, f'fold_{i}', 'images', 'val'),
            'nc': len(class_to_id),
            'names': {v:k for k,v in class_to_id.items()}
        }
        yaml_path = os.path.join(output_dir, f'fold_{i}', 'data.yaml')
        with open(yaml_path, 'w') as yaml_file:
            yaml.dump(data_yaml, yaml_file, default_flow_style=False)

def write_valid_test_csvs(train, test_path, cv_dir):
    folders = os.listdir(cv_dir)
    for folder in folders:
        if folder.startswith('fold_'):
            fold_path = os.path.join(cv_dir, folder)
            img_dir = os.path.join(fold_path, 'images', 'val')
            img_files = os.listdir(img_dir)
            mask = train['NAME'].isin(img_files)
            va = train.loc[mask].reset_index(drop=True)
            va['trustii_id'] = np.arange(va.shape[0])
            va.to_csv(os.path.join(fold_path, 'val.csv'), index=False)
            cmd = f'ln -s {test_path} {fold_path}/test.csv'
            os.system(cmd)

def save_bbox_images(img_folder, df):
    save_folder = img_folder.replace('images', 'bbox_images')
    os.makedirs(save_folder, exist_ok=True)
    for idx,row in tqdm(df.iterrows(), total=df.shape[0]):
        img_name = row['NAME']
        bbox = row['x1'],row['y1'],row['x2'],row['y2']
        save_path = os.path.join(save_folder, img_name.replace('.jpg', f'_{idx}.jpg'))
        image_path = os.path.join(img_folder, img_name)
        img = draw_bounding_box(image_path, bbox)
        img.save(save_path)

from pathlib import Path
def create_yolo_classification_folders(df, img_folder, det_cv_path, output_path):
    """
    Creates folders for YOLO classification format from a detection format input folder.

    Args:
        det_cv_path (str): Path to the input folder containing detection data in the "det_cv" format (e.g., "det_cv/fold_0/images/train").
        output_path (str, optional): Path to the output folder where the classification format folders will be created. Defaults to "cls_cv".
    """

    det_cv_path = Path(det_cv_path)
    output_path = Path(output_path)

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Iterate through folds
    for fold_dir in det_cv_path.glob("fold_*"):
        fold_num = fold_dir.name  # e.g., "fold_0"

        # Iterate through train and val
        for split in ["train", "val"]:
            images_dir = fold_dir / "images" / split
            labels_dir = fold_dir / "labels" / split

            if not images_dir.exists() or not labels_dir.exists():
                print(f"Warning: Missing images or labels directory for {fold_num}/{split}. Skipping.")
                continue

            # Create corresponding output directories
            output_fold_dir = output_path / fold_num / split
            output_fold_dir.mkdir(parents=True, exist_ok=True)

            # Iterate through label files
            img_names = os.listdir(str(images_dir))
            mask = df.NAME.isin(img_names)
            for _,row in tqdm(df[mask].iterrows(), total=df[mask].shape[0]):
                class_name = row['class']
                image_name = row['NAME']
                bbox_count = int(row['bbox_count'])
                class_count = int(row['class_count'])

                if bbox_count>1 and split == 'val':
                    continue

                if class_count>1:
                    continue
                image_file = os.path.join(img_folder, image_name)
                assert os.path.exists(image_file), f'{image_file} does not exist'

                class_dir = output_fold_dir / class_name
                class_dir.mkdir(parents=True, exist_ok=True)

                # Copy the image to the class directory
                destination_path = class_dir / image_name
                if not os.path.exists(destination_path):
                    cmd = f'ln -s {image_file} {destination_path}'
                    os.system(cmd)

        print(f"Successfully created classification folders for {fold_num}")

from glob import glob
def create_cls_cv_subset():
    cmd = f'cp -r {PATH}/cls_cv_more {PATH}/cls_cv_subset'
    os.system(cmd)

    rm = ['BA', 'LAM3', 'Thromb', 'Lysee', 'Er', 'PNN', 'EO']
    a = [glob(f'{PATH}/cls_cv_subset/fold_*/*/{i}') for i in rm]
    for i in a:
        for j in i:
            cmd = f'cd {j} && rm *.jpg'
            #print(cmd)
            os.system(cmd)
    a = glob(f'{PATH}/cls_cv_subset/fold_*/*.cache')
    for j in a:
        cmd = f'rm -rf {j}'
        os.system(cmd)

if __name__ == '__main__':
    df = pd.read_csv(f'{PATH}/train.csv')
    write_yolo_labels(df, f'{PATH}/images', f'{PATH}/labels')
    create_cv(df, f'{PATH}/images', f'{PATH}/labels', f'{PATH}/det_cv')
    write_valid_test_csvs(df, f'{PATH}/test.csv', f'{PATH}/det_cv')
    save_bbox_images(f'{PATH}/images', df)

    df['bbox_count'] = df.groupby('NAME')['NAME'].transform('count')
    df['class_count'] = df.groupby('NAME')['class'].transform('nunique')
    df = get_all_img_sizes(df,'{PATH}/images')
    create_yolo_classification_folders(df, f'{PATH}/images',
                                       f'{PATH}/det_cv',
                                       f'{PATH}/cls_cv_more')
