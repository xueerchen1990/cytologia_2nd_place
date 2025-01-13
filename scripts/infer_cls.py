import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")
import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from utils import get_yaml_value, random_flip, flip4
from PIL import Image
from focal_trainer import random_crop_with_bbox
from random import random
# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="yolo inference")

# Add arguments to the parser
parser.add_argument('--gpu', type=str, default='0', help='gpu id to run inference')
parser.add_argument('--data', type=str, choices=['test', 'val'], help='val or test')
parser.add_argument('--tag', type=str, help='tag of config')
parser.add_argument('--save_dir', type=str, default='./save', help='dir where weights are saved')
parser.add_argument('--data_dir', type=str, default=f'{PATH}/det_cv', help='dir where images live')
parser.add_argument('--fold', type=int, help='fold id')
parser.add_argument('--tte', type=int, default=0, help='tte repeat times')

# Parse the command-line arguments
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def predict(test, model, aug, img_dir, tte=1):
    cls_res = []
    probs = []
    for _,row in tqdm(test.iterrows(), total=test.shape[0]):
        img_name = row['NAME']
        bbox = row['x1'],row['y1'],row['x2'],row['y2']
        with Image.open(os.path.join(img_dir, img_name)) as img:
            pt = 0
            #imgs = flip4(img)
            imgs = []
            for _ in range(tte):
                # base=0.75
                # scale = base + random()*(1-base)
                # img = random_crop_with_bbox(img, bbox, scale)
                img = random_flip(img)
                imgs.append(img)
            if len(imgs)==0:
                imgs.append(img)
            res = model(imgs, verbose=False)
        for r in res:
            pt += r.probs.data.cpu().numpy()
        cls_res.append(res[0].names[pt.argmax()])
        scale = 1.0/len(imgs)
        probs.append(pt*scale)
    probs = np.array(probs)
    test['class'] = cls_res
    print(res[0].names)
    for k,v in res[0].names.items():
        test[v] = probs[:,k]
    return test

if __name__ == '__main__':
    # Define the path to the downloaded model
    model_dir = os.path.join(args.save_dir, args.tag, f'fold_{args.fold}')
    ckpt_path = os.path.join(model_dir, 'weights', "best.pt")

    # check if model eixsts
    if not os.path.exists(ckpt_path):
        print('Model not found. Exiting...')
        exit()
    # check if task is classification
    yaml_path = os.path.join(model_dir, 'args.yaml')
    task = get_yaml_value(yaml_path, 'task')

    if task!='classify':
        print('Task:', task)
        print('This script is only for classify tasks. Exiting...')
        exit()

    output = os.path.join(model_dir, f'{args.data}_res.csv')
    if os.path.exists(output):
        print('Output file already exists. Exiting...')
        exit()

    aug = get_yaml_value(yaml_path, 'augment')
    print(aug)
    cls_model = YOLO(ckpt_path)

    test = pd.read_csv(os.path.join('./save/aug1', f'fold_{args.fold}', f'{args.data}_res.csv'))
    # test = pd.read_csv(os.path.join(args.data_dir, f'fold_{args.fold}', 'val.csv'))
    img_dir = '/'.join(args.data_dir.split('/')[:-1]+['images'])
    print(img_dir)

    sub = predict(test, cls_model, aug, img_dir, args.tte)
    sub.to_csv(output, index=False)