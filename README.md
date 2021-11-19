# easy-deep-learning-pytorch

# install conda

# Set up experimental env
conda create -p ./easydl-env python=3.8
conda activate ./easydl-env
conda deactivate


# check out specific version
export codedir="easydl-v1.3"

git clone https://github.com/xiaonanzzz/easy-deep-learning-pytorch.git $codedir
git -C $codedir fetch; git -C $codedir checkout 11618a4


pip install -r $codedir/requirements.txt

## run cifar test
export PYTHONPATH=./easydl-v1.3:$PYTHONPATH;
python3 -m easydl.example.cifar --wandb_key $wandb_key --lr 0.005 --function train_simple_net

