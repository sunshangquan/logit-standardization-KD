# Use this script to train your own student model.

# PKT
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill pkt --model_s resnet20 -r 0.1 -a 0.9 -b 30000
# SP
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill similarity --model_s resnet20 -r 0.1 -a 0.9 -b 3000
# VID
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill vid --model_s resnet20 -r 0.1 -a 0.9 -b 1
# CRD
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet20 -r 0.1 -a 0.9 -b 0.8
# SRRL
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill srrl --model_s resnet20 -r 0.1 -a 0.9 -b 1
# DKD
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill dkd --model_s resnet20 -r 1 -a 0 -b 1 --dkd_alpha 1 --dkd_beta 2


python3 train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth \
        --distill kd \
        --model_s resnet20 -r 0.1 -a 9 -b 0 --kd_T 2 \
        --batch_size 64 --learning_rate 0.05 \
        --have_mlp 1 --mlp_name 'global'\
        --cosine_decay 1 --decay_max 0 --decay_min -1 --decay_loops 10 \
        --save_model \
        --experiments_dir 'tea-res56-stu-res20/kd/global_T/your_experiment_name' \
        --experiments_name 'fold-1'
        
