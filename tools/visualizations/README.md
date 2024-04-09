
## Reproduce Figure 3, 4, 5

You could download model weights from [visualization_weights.zip](<https://github.com/sunshangquan/logit-standardization-KD/releases/tag/visualization_weights>) or use your own.

Replace the names in the two jupyter notebooks:

```
"/home/ssq/Desktop/phd/KD/CTKD-main/save/student_model0/tea-res32x4-stu-res8x4/kd/fold-1/resnet8x4_best.pth": ctkd_0_resnet8x4_best.pth

"/home/ssq/Desktop/phd/KD/CTKD-main/save/student_model/tea-res32x4-stu-res8x4/kd_ours/fold-1/resnet8x4_best.pth": ctkd_1_resnet8x4_best.pth

"/home/ssq/Desktop/phd/KD/mdistiller-master/output/cifar100_baselines_0/kd,res32x4,res8x4/student_best": kd_0_resnet8x4_best.pth

"/home/ssq/Desktop/phd/KD/mdistiller-master/output/cifar100_baselines/kd,res32x4,res8x4/student_best": kd_1_resnet8x4_best.pth

"/home/ssq/Desktop/phd/KD/mdistiller-master/output/cifar100_baselines_0/dkd,res32x4,res8x4/student_best": dkd_0_resnet8x4_best.pth

"/home/ssq/Desktop/phd/KD/mdistiller-master/output/cifar100_baselines/dkd,res32x4,res8x4/student_best": dkd_1_resnet8x4_best.pth

"/home/ssq/Desktop/phd/KD/Multi-Level-Logit-Distillation-main/output/cifar100_baselines/kd_ours,res32x4,res8x4/student_best": mlkd_0_resnet8x4_best.pth

"/home/ssq/Desktop/phd/KD/Multi-Level-Logit-Distillation-main/output/cifar100_baselines/kd_ours,res32x4,res8x4_1/student_best": mlkd_1_resnet8x4_best.pth

"/home/ssq/Desktop/phd/KD/mdistiller-master/download_ckpts/cifar_teachers/resnet32x4_vanilla/ckpt_epoch_240.pth": teacher_resnet32x4_best.pth

"/home/ssq/Desktop/phd/KD/mdistiller-master/output/ctkd/kd/resnet8x4_best.pth": ctkd_resnet8x4_best.pth
```

Run the two jupyter notebooks.

```
# Figure 3 and 5
jupyter notebook correlation.ipynb
# Figure 4
jupyter notebook tsne.ipynb
```