#!/bin/bash

## Schedule jobs ##

## REMEMBER TO ACTIVATE ENVIRONMENT ##

# TRAIN 1
# RQ1
python3 imitate_episodes_record_torque.py \
--task_name lego_insert \
--ckpt_dir ./ckpt_dir/lego_insert \
--robot /home/khw/interbotix_ws/src/aloha/config/robot/aloha_solo \
--policy_class ACT \
--kl_weight 10 \
--chunk_size 100 \
--hidden_dim 512 \
--batch_size 16 \
--dim_feedforward 3200 \
--num_epochs 6000 \
--lr 2e-5 --seed 0 \
> lego_insert.log 2>&1

sleep 30  # Increased cooldown to ensure file handles are closed and GPU is clear

# LEGO-INSERT
# TRAIN 2
python3 imitate_episodes_record_torque.py \
--task_name lego_insert \
--ckpt_dir ./ckpt_dir/lego_insert_obs_target \
--robot /home/khw/interbotix_ws/src/aloha/config/robot/aloha_solo \
--policy_class ACT \
--kl_weight 10 \
--chunk_size 100 \
--hidden_dim 512 \
--batch_size 16 \
--dim_feedforward 3200 \
--num_epochs 8000 \
--lr 2e-5 --seed 0 --use_obs_target \
> lego_insert_obs_target.log 2>&1
sleep 30

#######################
##### BOARD WIPE ######

#1
python3 imitate_episodes_record_torque.py \
--task_name board_wipe \
--ckpt_dir ./ckpt_dir/board_wipe \
--robot /home/khw/interbotix_ws/src/aloha/config/robot/aloha_solo \
--policy_class ACT \
--kl_weight 10 \
--chunk_size 100 \
--hidden_dim 512 \
--batch_size 16 \
--dim_feedforward 3200 \
--num_epochs 6000 \
--lr 2e-5 --seed 0 \
> board_wipe.log 2>&1

sleep 30

# 2
python3 imitate_episodes_record_torque.py \
--task_name board_wipe \
--ckpt_dir ./ckpt_dir/board_wipe_obs_target \
--robot /home/khw/interbotix_ws/src/aloha/config/robot/aloha_solo \
--policy_class ACT \
--kl_weight 10 \
--chunk_size 100 \
--hidden_dim 512 \
--batch_size 16 \
--dim_feedforward 3200 \
--num_epochs 8000 \
--lr 2e-5 --seed 0 --use_obs_target \
> board_wipe_obs_target.log 2>&1

sleep 30

# 3
python3 imitate_episodes_record_torque.py \
--task_name board_wipe \
--ckpt_dir ./ckpt_dir/board_wipe_2view \
--robot /home/khw/interbotix_ws/src/aloha/config/robot/aloha_solo \
--policy_class ACT \
--kl_weight 10 \
--chunk_size 100 \
--hidden_dim 512 \
--batch_size 16 \
--dim_feedforward 3200 \
--num_epochs 8000 \
--exclude_cameras camera_left_shoulder \
--lr 2e-5 --seed 0 \
> board_wipe_2view.log 2>&1

sleep 30

#4
python3 imitate_episodes_record_torque.py \
--task_name board_wipe \
--ckpt_dir ./ckpt_dir/board_wipe_2view_obs_target \
--robot /home/khw/interbotix_ws/src/aloha/config/robot/aloha_solo \
--policy_class ACT \
--kl_weight 10 \
--chunk_size 100 \
--hidden_dim 512 \
--batch_size 16 \
--dim_feedforward 3200 \
--num_epochs 8000 \
--exclude_cameras camera_left_shoulder \
--lr 2e-5 --seed 0 --use_obs_target \
> board_wipe_2view_obs_target.log 2>&1

sleep 30

#5
python3 imitate_episodes_record_torque.py \
--task_name board_wipe \
--ckpt_dir ./ckpt_dir/board_wipe_1view \
--robot /home/khw/interbotix_ws/src/aloha/config/robot/aloha_solo \
--policy_class ACT \
--kl_weight 10 \
--chunk_size 100 \
--hidden_dim 512 \
--batch_size 16 \
--dim_feedforward 3200 \
--num_epochs 6000 \
--exclude_cameras camera_right_shoulder camera_left_shoulder \
--lr 2e-5 --seed 0 \
> board_wipe_1view.log 2>&1

sleep 30

#6
python3 imitate_episodes_record_torque.py \
--task_name board_wipe \
--ckpt_dir ./ckpt_dir/board_wipe_1view_obs_target \
--robot /home/khw/interbotix_ws/src/aloha/config/robot/aloha_solo \
--policy_class ACT \
--kl_weight 10 \
--chunk_size 100 \
--hidden_dim 512 \
--batch_size 16 \
--dim_feedforward 3200 \
--num_epochs 6000 \
--exclude_cameras camera_right_shoulder camera_left_shoulder \
--lr 2e-5 --seed 0 --use_obs_target \
> board_wipe_1view_obs_target.log 2>&1


## TRAINING COMPLETED ##
echo "Full training sequence complete at $(date)"
