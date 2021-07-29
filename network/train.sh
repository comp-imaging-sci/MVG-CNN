#!/bin/bash 
timelen=15
test_mice=200128
echo "$test_mice"
data_path=/shared/anastasio-s1/MRI/xiaohui/mouse_optical/sleep-stage/
mice_flist=config14
dataset=dataloader_MVG
model=model_cnn2d
num_classes=3
brain_area=whole
mode=test
lr_init=0.0001
gamma=2
num_epochs=100
batch_size=4
seed=42
loss=categorical_focal
gradcam_label=2
model_savedir=/shared/anastasio-s1/MRI/xiaohui/mouse_optical/sleep-stage/paper/final_cps/2020_${brain_area}_lr${lr_init}_epochs${num_epochs}_class${num_classes}_batch4_nodrop_32_64_128_gap_mouse11_${timelen}s_focal_gamma${gamma}_${mice_flist}_alpha0.25
echo "$model_savedir"

python main.py --model $model --dataset $dataset --mode $mode --data_path $data_path --mice_flist $mice_flist --num_epochs $num_epochs --brain_area $brain_area --model_savedir $model_savedir --batch_size $batch_size --lr_init $lr_init --num_epochs $num_epochs --num_classes $num_classes --seed $seed --gradcam_label $gradcam_label --loss $loss --test_mice $test_mice --gamma $gamma --timelen $timelen

