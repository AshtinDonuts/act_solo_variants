
### Collecting evaluation demos
If using dynamic torques:

Launch the corresponding dynamic torques node...
```
ros2 launch aloha aloha_bringup.launch.py robot:=aloha_solo get_dynamics_torque:=true
```

Use the corresponding record torque script that enables recording dynamics torque.
```
python3 imitate_episodes_rec_torque.py   --task_name plug_insert   --ckpt_dir ./ckpt_dir/plug_insert_base60   --robot /home/khw/interbotix_ws/src/aloha/config/robot/aloha_solo   --policy_class ACT   --kl_weight 10   --chunk_size 100   --hidden_dim 512   --batch_size 16   --dim_feedforward 3200   --num_epochs 8000   --lr 1e-5 --seed 0   --eval   --temporal_agg
```

To Shutdown
```
python3 sleep.py -r aloha_solo
```
Then close the Launch terminal.
