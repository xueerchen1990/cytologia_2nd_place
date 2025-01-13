python train_det.py --gpu 0 --fold 0 --model yolov9c.pt --tag aug1 
python train_det.py --gpu 0 --fold 1 --model yolov9c.pt --tag aug1
python train_det.py --gpu 0 --fold 1 --model yolov9c.pt --tag aug1
python train_det.py --gpu 0 --fold 3 --model yolov9c.pt --tag aug1

python train_cls_rotate50.py --model yolo11l-cls.pt --tag rotate50 --gpu 0 --fold 0
python train_cls_rotate50.py --model yolo11l-cls.pt --tag rotate50 --gpu 0 --fold 1
python train_cls_rotate50.py --model yolo11l-cls.pt --tag rotate50 --gpu 0 --fold 2
python train_cls_rotate50.py --model yolo11l-cls.pt --tag rotate50 --gpu 0 --fold 3

python train_cls_rotate.py --model yolo11l-cls.pt --tag rotate --gpu 0 --fold 0
python train_cls_rotate.py --model yolo11l-cls.pt --tag rotate --gpu 0 --fold 1
python train_cls_rotate.py --model yolo11l-cls.pt --tag rotate --gpu 0 --fold 2
python train_cls_rotate.py --model yolo11l-cls.pt --tag rotate --gpu 0 --fold 3

python train_cls_crop.py --model yolo11l-cls.pt --tag crop --gpu 0 --fold 0
python train_cls_crop.py --model yolo11l-cls.pt --tag crop --gpu 0 --fold 1
python train_cls_crop.py --model yolo11l-cls.pt --tag crop --gpu 0 --fold 2
python train_cls_crop.py --model yolo11l-cls.pt --tag crop --gpu 0 --fold 3
