import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")
import os
import argparse
import pandas as pd

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="yolo inference")

# Add arguments to the parser
parser.add_argument('--gpu', type=str, default='0', help='gpu id to run inference')
parser.add_argument('--data', type=str, choices=['test', 'val'], help='val or test')
parser.add_argument('--tag', type=str, help='tag of config')
parser.add_argument('--save_dir', type=str, help='dir where weights are saved')
parser.add_argument('--data_dir', type=str, help='dir where images live')
parser.add_argument('--fold', type=int, help='fold id')
#parser.add_argument('--cls', type=str, help='classification model name')

# Parse the command-line arguments
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import torchvision
import numpy as np
from ultralytics.utils import LOGGER
from tqdm import tqdm

from PIL import Image
import time
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops

from utils import get_yaml_value

def my_non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels.
        in_place (bool): If True, the input prediction tensor will be modified in place.
        rotated (bool): If Oriented Bounding Boxes (OBB) are being passed for NMS.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks + num_classes) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ..., all class probabilities).
    """

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        if in_place:
            prediction[..., :4] = ops.xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        else:
            prediction = torch.cat((ops.xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm + nc), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = ops.xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)
        # Store all class probabilities
        all_cls_probs = cls

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = ops.nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = torch.cat((x[i], all_cls_probs[i]), dim=1)

        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output



class MyCustomPredictor(DetectionPredictor):

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = my_non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        probs = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            pred,prob = pred[:,:-23], pred[:,-23:]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
            if len(prob):
                probs.append(prob.cpu().numpy()[0])
        self.probs.extend(probs)
        return results

def predict(test, model, aug, img_dir):

    tids = test.groupby('NAME')['trustii_id'].agg(list).to_dict()
    df = test.groupby('NAME').agg({'trustii_id':'count'})
    df.columns = ['bbox_count']
    df = df.reset_index()
    
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        img_path = os.path.join(img_dir, row['NAME'])
        break
    img = Image.open(img_path)
    res = model(img,verbose=False, augment=aug)
    my_predictor = MyCustomPredictor()
    my_predictor.args = model.predictor.args
    my_predictor.probs = []
    model.predictor = my_predictor
    
    def dec(x):
        return [float(f'{i:.3f}') for i in x]
    outs = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        img_path = os.path.join(img_dir, row['NAME'])
        results = model(img_path, max_det=row['bbox_count']+1, verbose=False, augment=aug)[0]
        xyxys = results.boxes.xyxy.cpu().numpy().tolist()
        clss = results.boxes.cls.cpu().numpy().tolist()
        ids = tids[row['NAME']]
        
        for c,idx in enumerate(ids):
            if c < len(clss):
                x1,y1,x2,y2 = dec(xyxys[c])
                cls_id = int(clss[c])
                cls = results.names[cls_id]
            outs.append([idx, row['NAME'], x1, y1, x2, y2, cls])
            if len(outs) == len(my_predictor.probs)+1:
                tmp = np.zeros(23)
                tmp[cls_id] = 1
                my_predictor.probs.append(tmp)
            elif len(outs) != len(my_predictor.probs):
                assert 0
    sub = pd.DataFrame(outs, columns = ['trustii_id', 'NAME', 'x1', 'y1', 'x2', 'y2', 'class'])
    #subx = test[['trustii_id']].merge(sub, on='trustii_id', how='left')
    probs = np.array(my_predictor.probs)
    return sub, probs, res[0].names


if __name__ == '__main__':
    # Define the path to the downloaded model
    model_dir = os.path.join(args.save_dir, args.tag, f'fold_{args.fold}')
    ckpt_path = os.path.join(model_dir, 'weights', "best.pt")
    # check if model eixsts
    if not os.path.exists(ckpt_path):
        print('Model not found. Exiting...')
        exit()
    # check if task is detection
    yaml_path = os.path.join(model_dir, 'args.yaml')
    task = get_yaml_value(yaml_path, 'task')
    if task!='detect':
        print('Task:', task)
        print('This script is only for detection tasks. Exiting...')
        exit()

    output = os.path.join(model_dir, f'{args.data}_res.csv')
    if os.path.exists(output):
        print('Output file already exists. Exiting...')
        exit()
    aug = get_yaml_value(yaml_path, 'augment')
    print(aug)
    det_model = YOLO(ckpt_path)

    test = pd.read_csv(os.path.join(args.data_dir, f'fold_{args.fold}', f'{args.data}.csv'))
    #img_dir = os.path.join(args.data_dir, f'fold_{args.fold}', 'images', f'{args.data}')
    img_dir = '/'.join(args.data_dir.split('/')[:-1]+['images'])
    print(img_dir)
    # get predictions
    sub, probs, names = predict(test, det_model, aug, img_dir)
    print(names)
    for k,v in names.items():
        sub[v] = probs[:,k]
        # only keep 5 decimal places
        sub[v] = sub[v].round(5)
    sub = test[['trustii_id']].merge(sub, on='trustii_id', how='left')
    sub.to_csv(output, index=False)