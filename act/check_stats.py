import os, pickle, numpy as np
ckpt_dir = '/home/khw/act_training_evaluation/act/ckpt_dir/picknplace_STATIONARY_1'  # change this path if needed
path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
with open(path, 'rb') as f: stats = pickle.load(f)
am, as_ = stats['action_mean'], stats['action_std']
print("\nIndex | mean      | std       \n-----------------------------")
for i, (m, s) in enumerate(zip(am, as_)):
    print(f"{i:5d} | {m:8.4f} | {s:8.4f}")