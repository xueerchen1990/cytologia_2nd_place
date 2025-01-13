python infer_cls.py --data test --tag rotate50 --gpu 0 --fold 0
python infer_cls.py --data test --tag rotate50 --gpu 0 --fold 1
python infer_cls.py --data test --tag rotate50 --gpu 0 --fold 2
python infer_cls.py --data test --tag rotate50 --gpu 0 --fold 3

python infer_cls.py --data test --tag rotate --gpu 0 --fold 0
python infer_cls.py --data test --tag rotate --gpu 0 --fold 1
python infer_cls.py --data test --tag rotate --gpu 0 --fold 2
python infer_cls.py --data test --tag rotate --gpu 0 --fold 3

python infer_cls.py --data test --tag crop --gpu 0 --fold 0
python infer_cls.py --data test --tag crop --gpu 0 --fold 1
python infer_cls.py --data test --tag crop --gpu 0 --fold 2
python infer_cls.py --data test --tag crop --gpu 0 --fold 3

python infer.py --gpu 0 --fold 0 --data test --tag aug1 
python infer.py --gpu 0 --fold 1 --data test --tag aug1
python infer.py --gpu 0 --fold 1 --data test --tag aug1
python infer.py --gpu 0 --fold 3 --data test --tag aug1
