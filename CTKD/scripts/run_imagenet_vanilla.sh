# Use this script to train your own teacher model.

python3 train_teacher.py --model ResNet18 \
                --dataset 'imagenet' \
                --multiprocessing-distributed \
                --batch_size 256 \
                --epochs 100 \
                --learning_rate 0.1 \
                --lr_decay_epochs '30,60,90' \
                --lr_decay_rate 0.1 \
                --weight_decay 1e-4 \
                --experiments_dir 'baseline/imagenet/resnet18' \
                --experiments_name 'fold-1'
