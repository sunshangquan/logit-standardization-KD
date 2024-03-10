# Use this script to train your own student model.

# PKT
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill pkt --model_s resnet20 -r 1 -a 1 -b 30000
# SP
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill similarity --model_s resnet20 -r 1 -a 1 -b 3000
# VID
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill vid --model_s resnet20 -r 1 -a 1 -b 1
# CRD
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet20 -r 1 -a 1 -b 0.8
# SRRL
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill srrl --model_s resnet20 -r 1 -a 1 -b 1
# DKD
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill srrl --model_s resnet20 -r 1 -a 0 -b 1 --dkd_alpha 1 --dkd_beta 1


python3 train_student.py --path-t ./save/models/ResNet34/resnet34.pth \
        --distill kd \
        --batch_size 256 --epochs 120 --dataset imagenet \
        --gpu_id 0,1,2,3,4,5,6,7 --num_workers 32 \
        --multiprocessing-distributed --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
        --model_s ResNet18 -r 1 -a 2 -b 0 --kd_T 9 \
        --have_mlp 1 --mlp_name 'global' \
        --t_start 1 --t_end 20 --cosine_decay 1 \ 
        --decay_max 0 --decay_min -1 --decay_loops 5 \
        --save_model \
        --experiments_dir 'imagenet-tea-res34-stu-res18/ctkd_120/your_experiment_name' \
        --experiments_name 'fold-1'
            