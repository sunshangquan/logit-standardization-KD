# Use this script to train your own teacher model.

python3 train_teacher.py --model resnet56 \
                --batch_size 64 \
                --epochs 240 \
                --learning_rate 0.05 \
                --lr_decay_epochs '150,180,210' \
                --lr_decay_rate 0.1 \
                --experiments_dir 'baseline/resnet56' \
                --experiments_name 'fold-1' \