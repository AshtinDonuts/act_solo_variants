import os, pickle, numpy as np
ckpt_dir = '/home/khw/act_training_evaluation/act/ckpt_dir/ID01_picknplace'  # change this path if needed
path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
with open(path, 'rb') as f: stats = pickle.load(f)
# Read and print means and stds for action, effort, and qpos
entries = [
    ('action', stats['action_mean'], stats['action_std']),
    ('effort', stats['effort_mean'], stats['effort_std']),
    ('qpos',   stats['qpos_mean'],   stats['qpos_std']),
]

for name, mean_vals, std_vals in entries:
    print(f"\n{name.upper()} STATS")
    print("Index | mean      | std       \n-----------------------------")
    for i, (m, s) in enumerate(zip(mean_vals, std_vals)):
        print(f"{i:5d} | {m:8.4f} | {s:8.4f}")