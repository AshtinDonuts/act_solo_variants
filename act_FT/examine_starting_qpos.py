#!/usr/bin/env python3
"""
Script to examine the starting joint state from qpos pickle file.
"""

import pickle
import numpy as np
import os

# Path to the qpos pickle file
qpos_file = '/home/khw/ACT_old/act/ckpt_dir/lego_insert_feb8/qpos_0.pkl'

def examine_starting_qpos(qpos_file):
    """Load and examine the starting joint state from qpos pickle file."""
    
    if not os.path.exists(qpos_file):
        print(f"Error: File not found: {qpos_file}")
        return
    
    print(f"Loading qpos data from: {qpos_file}\n")
    
    # Load the pickle file
    with open(qpos_file, 'rb') as f:
        data = pickle.load(f)
    
    # Extract data
    qpos_list = data.get('qpos', [])
    target_qpos_list = data.get('target_qpos', [])
    rewards = data.get('rewards', [])
    
    print("=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"Number of timesteps: {len(qpos_list)}")
    print(f"Number of target_qpos: {len(target_qpos_list)}")
    print(f"Number of rewards: {len(rewards)}")
    print()
    
    if len(qpos_list) == 0:
        print("Error: qpos_list is empty!")
        return
    
    # Get starting joint state (first timestep)
    starting_qpos = qpos_list[0]
    
    print("=" * 80)
    print("STARTING JOINT STATE (qpos[0])")
    print("=" * 80)
    
    if isinstance(starting_qpos, np.ndarray):
        print(f"Shape: {starting_qpos.shape}")
        print(f"Data type: {starting_qpos.dtype}")
        print()
        print("Joint values:")
        for i, val in enumerate(starting_qpos):
            print(f"  Joint {i:2d}: {val:10.6f}")
        print()
        print(f"Full array: {starting_qpos}")
    else:
        print(f"Type: {type(starting_qpos)}")
        print(f"Value: {starting_qpos}")
    
    print()
    print("=" * 80)
    print("COMPARISON: First 5 timesteps")
    print("=" * 80)
    for i in range(min(5, len(qpos_list))):
        qpos = qpos_list[i]
        if isinstance(qpos, np.ndarray):
            print(f"Timestep {i}: {qpos}")
        else:
            print(f"Timestep {i}: {qpos}")
    
    print()
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    if len(qpos_list) > 0 and isinstance(qpos_list[0], np.ndarray):
        all_qpos = np.array(qpos_list)
        print(f"All qpos shape: {all_qpos.shape}")
        print(f"Mean qpos: {np.mean(all_qpos, axis=0)}")
        print(f"Std qpos:  {np.std(all_qpos, axis=0)}")
        print(f"Min qpos:  {np.min(all_qpos, axis=0)}")
        print(f"Max qpos:  {np.max(all_qpos, axis=0)}")
        print()
        print("Starting qpos vs Mean:")
        diff = starting_qpos - np.mean(all_qpos, axis=0)
        print(f"Difference: {diff}")
        print(f"Max absolute difference: {np.max(np.abs(diff)):.6f}")
    
    print()
    print("=" * 80)
    print("REWARDS SUMMARY")
    print("=" * 80)
    if len(rewards) > 0:
        rewards_array = np.array([r for r in rewards if r is not None])
        if len(rewards_array) > 0:
            print(f"Total rewards: {len(rewards_array)}")
            print(f"Sum of rewards: {np.sum(rewards_array):.2f}")
            print(f"Mean reward: {np.mean(rewards_array):.4f}")
            print(f"Max reward: {np.max(rewards_array):.4f}")
            print(f"Min reward: {np.min(rewards_array):.4f}")
            print(f"First reward: {rewards[0]}")
            print(f"Last reward: {rewards[-1]}")

if __name__ == '__main__':
    examine_starting_qpos(qpos_file)
