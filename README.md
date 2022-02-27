# easy-deep-learning-pytorch


# How to use it? 

## in jupyter note book

```
# checkout the latest version
import sys

codedir = 'easy-dl-repo/main'
!git clone https://github.com/xiaonanzzz/easy-deep-learning-pytorch.git $codedir
!git -C $codedir fetch; git -C $codedir pull
sys.path.append(codedir)
```

```ipython
# checkout the latest version
import sys
ver = '084689d'
codedir = 'easy-dl-repo/' + ver
!git clone https://github.com/xiaonanzzz/easy-deep-learning-pytorch.git $codedir
!git -C $codedir fetch; git -C $codedir checkout $ver
sys.path.append(codedir)
```

```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install timm

```

# install conda

## Set up experimental env
conda create -p ./easydl-env python=3.8
conda activate ./easydl-env
conda deactivate


## check out specific version
export codedir="easydl-v1.3"

git clone https://github.com/xiaonanzzz/easy-deep-learning-pytorch.git $codedir
git -C $codedir fetch; git -C $codedir checkout 11618a4


pip install -r $codedir/requirements.txt

## run cifar test
```shell
export PYTHONPATH=./easydl-v1.3:$PYTHONPATH;
python3 -m easydl.example.cifar --wandb_key $wandb_key --lr 0.005 --function train_simple_net
## resnet on cifar
export PYTHONPATH="easydl-dev"; \
echo $PYTHONPATH; \
python3 -m easydl.example.cifar --wandb_key $wandb_key --function train_resnet_18
```



