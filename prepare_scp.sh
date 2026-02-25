# Transfer from Local to Server
scp -r /mnt/c2d9b23a-b03e-4fdb-82ad-59f039ec9e3e/khw/board_wipe 10.12.65.153:/home/khw/aloha_data/board_wipe

# Transfer from Server to Local
scp -v -r 10.12.65.153:/home/khw/act_solo_variants/act/ckpt_dir/board_wipe_chunk50 /home/khw/ACT_old/act/ckpt_dir
