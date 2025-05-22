# iGAT

## Training

iGAT ADP CIFAR10
```Python
python -u train/train_igat_adp_cifar10_1.py --model-num 8 --num-classes 10 --lr 0.001 --plus-adv --data-dir "../data/" --epochs 2000 --batch-size 512 --model-file ""
```

iGAT DVERGE CIFAR10
```Python
python -u train/train_igat_dverge_cifar10.py --model-num 8 --num-classes 10 --lr 0.001 --plus-adv --data-dir "../data/" --epochs 2000 --batch-size 512 --model-file "" 
```

iGAT DVERGE CIFAR100
```Python
python -u train/train_igat_dverge_cifar100.py --model-num 8 --num-classes 100 --lr 0.1 --plus-adv --data-dir "../data/" --epochs 2000 --batch-size 512 --model-file ""
```

## Testing
PGD
```Python
python -u test_PGD.py --model-file "./ckpts/iGAT_ADP_cifar10_1.pth" --leaky-relu 1
```
CW
```Python
python -u test_CW.py --model-file "./ckpts/iGAT_ADP_cifar10_2.pth" --leaky-relu 1
```

Sign Attack
```Python
cd BlackboxBench-master
python attack_cifar100.py ensemble_sign_linf_config.json
```

AutoAttack
```Python
python -u test_AA.py --model-file "./ckpts/iGAT_DVERGE_cifar10_1.pth" --leaky-relu 0
```
