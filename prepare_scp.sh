# Transfer from Local to Server
scp -r /mnt/c2d9b23a-b03e-4fdb-82ad-59f039ec9e3e/khw/foam_stop 10.12.65.153:/home/khw/aloha_data/foam_stop

# Transfer from Server to Local
scp -v -r 10.12.65.153:/home/khw/IROS_project/act_solo_variants/act/ckpt_dir/d18/ /home/khw/ACT_old/act/ckpt_dir/d18/

scp -v -r 10.12.65.153:/home/khw/IROS_project/act_solo_torque/act_screw_torque/ckpt_dir/i14/ /home/khw/ACT_screw_torque/act_screw_torque/ckpt_dir/i14/
