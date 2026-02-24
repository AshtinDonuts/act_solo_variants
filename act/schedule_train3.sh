#!/bin/bash

## Schedule jobs ##

## REMEMBER TO ACTIVATE ENVIRONMENT ##

# TRAIN 1
# Model: ACT Base
# Cam Views: 3 views
python3 imitate_episodes_record_torque.py \
--task_name lego_insert_h \
--ckpt_dir ./ckpt_dir/lego_insert_h \
--robot /home/khw/interbotix_ws/src/aloha/config/robot/aloha_solo \
--policy_class ACT \
--kl_weight 10 \
--chunk_size 100 \
--hidden_dim 512 \
--batch_size 16 \
--dim_feedforward 3200 \
--num_epochs 8000 \
--lr 2e-5 --seed 0 \
> lego_insert_h.log 2>&1

sleep 30  # Increased cooldown to ensure file handles are closed and GPU is clear

# TRAIN 2
# Model: ACT Base - Obs joints as Target
# Cam Views: 3 views
python3 imitate_episodes_record_torque.py \
--task_name lego_insert_h \
--ckpt_dir ./ckpt_dir/lego_insert_h_obs_target \
--robot /home/khw/interbotix_ws/src/aloha/config/robot/aloha_solo \
--policy_class ACT \
--kl_weight 10 \
--chunk_size 100 \
--hidden_dim 512 \
--batch_size 16 \
--dim_feedforward 3200 \
--num_epochs 8000 \
--lr 2e-5 --seed 0 --use_obs_target \
> lego_insert_h_obs_target.log 2>&1

sleep 30

# TRAIN 3
# Model: ACT Base
# Cam Views: 2 views (right shoulder, wrist)
python3 imitate_episodes_record_torque.py \
--task_name lego_insert_h \
--ckpt_dir ./ckpt_dir/lego_insert_h_2view \
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
> lego_insert_h_2view.log 2>&1

sleep 30

# TRAIN 4
# Model: ACT Base - Obs joints as Target
# Cam Views: 2 views (right shoulder, wrist)
python3 imitate_episodes_record_torque.py \
--task_name lego_insert_h \
--ckpt_dir ./ckpt_dir/lego_insert_h_2view_obs_target \
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
> lego_insert_h_2view_obs_target.log 2>&1

sleep 30

# TRAIN 5
# Model: ACT Base
# Cam Views: 1 view (wrist)
python3 imitate_episodes_record_torque.py \
--task_name lego_insert_h \
--ckpt_dir ./ckpt_dir/lego_insert_h_1view \
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
> lego_insert_h_1view.log 2>&1

sleep 30

# TRAIN 6
# Model: ACT Base - Obs joints as Target
# Cam Views: 1 view (wrist)
python3 imitate_episodes_record_torque.py \
--task_name lego_insert_h \
--ckpt_dir ./ckpt_dir/lego_insert_h_1view_obs_target \
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
> lego_insert_h_1view_obs_target.log 2>&1


## TRAINING COMPLETED ##
echo "Full training sequence complete at $(date)"
