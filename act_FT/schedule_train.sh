#!/bin/bash

## Schedule jobs ##

# 1. Finish off 30 episode training
echo "Starting Run 1: 30 episodes, no left camera..."
python3 imitate_episodes.py \
  --task_name plug_insert \
  --ckpt_dir ./ckpt_dir/plug_insert_base_no_left \
  --robot /home/khw/interbotix_ws/src/aloha/config/robot/aloha_solo \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --batch_size 16 \
  --dim_feedforward 3200 \
  --num_epochs 6000 \
  --lr 1e-5 --seed 0 \
  --exclude_cameras camera_left_shoulder \
  > plug_insert_base_no_left.log 2>&1 

# 2. Combine the data
echo "Combining datasets..."
mv /mnt/c2d9b23a-b03e-4fdb-82ad-59f039ec9e3e/khw/plug_insert2/*.hdf5 /mnt/c2d9b23a-b03e-4fdb-82ad-59f039ec9e3e/khw/plug_insert/

sleep 30  # Increased cooldown to ensure file handles are closed and GPU is clear

# 3. ALL CAMS (60 episodes)
echo "Starting Run 2: 60 episodes, all cameras..."
python3 imitate_episodes.py \
  --task_name plug_insert \
  --ckpt_dir ./ckpt_dir/plug_insert_base60 \
  --robot /home/khw/interbotix_ws/src/aloha/config/robot/aloha_solo \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --batch_size 16 \
  --dim_feedforward 3200 \
  --num_epochs 8000 \
  --lr 1e-5 --seed 0 \
  > plug_insert_base60.log 2>&1

sleep 30

# 4. Remove Left Shoulder (60 episodes)
echo "Starting Run 3: 60 episodes, no left camera..."
python3 imitate_episodes.py \
  --task_name plug_insert \
  --ckpt_dir ./ckpt_dir/plug_insert_base60_no_left \
  --robot /home/khw/interbotix_ws/src/aloha/config/robot/aloha_solo \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --batch_size 16 \
  --dim_feedforward 3200 \
  --num_epochs 8000 \
  --lr 1e-5 --seed 0 \
  --exclude_cameras camera_left_shoulder \
  > plug_insert_base60_no_left.log 2>&1

## TRAINING COMPLETED ##
echo "Full training sequence complete at $(date)"