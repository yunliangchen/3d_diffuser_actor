"""Main script for trajectory optimization."""
import os
import random
import numpy as np
import torch
import tap
from pathlib import Path
from typing import Tuple, Optional
from engine import BaseTrainTester
import torch.distributed as dist
from datasets.dataset_droid_sparse import RLBenchDataset
from tqdm import tqdm

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


if __name__ == '__main__':
    train_dataset = RLBenchDataset(
            root="/lustre/fsw/portfolios/nvr/users/lawchen/droid_raw/droid_raw/droid_with_pcd_20k",
            cache_size=0,
            num_iters=None,
            max_episode_length=1000,
            training=True,
            image_rescale=(
                1.0, 1.0
            ),
            return_low_lvl_trajectory=True,
            dense_interpolation=True,
            interpolation_length=20,
            use_wrist_camera=False,
        )
    """
    ret_dict = {
            "task": ["dummy" for _ in frame_ids],
            "rgbs": rgbs,  # e.g. tensor (n_frames, n_cam, 3, H, W)
            "pcds": pcds,  # e.g. tensor (n_frames, n_cam, 3, H, W)
            "action": action,  # e.g. tensor (n_frames, 8), target pose
            "instr": instr,  # a (n_frames, 53, 512) tensor
            "curr_gripper": gripper, # e.g. tensor (n_frames, 8)
            "curr_gripper_history": gripper_history, (n_frames, nhist, 8)
            "trajectory": traj,  # e.g. tensor (n_frames, T, 8)
            "trajectory_mask": traj_mask.bool()  # tensor (n_frames, T)
        }
    """
    lens = []
    trajectories_xyz = []
    indices = np.random.permutation(len(train_dataset))
    for i in tqdm(indices[:10]):
        item = train_dataset.__getitem__(i)
        lens.append(item["rgbs"].shape[0])
        trajectories_xyz.append(item["trajectory"][:, :, :3])
        
        num_key_poses = item["rgbs"].shape[0]
        if num_key_poses >= 3:
            key_poses = item["rgbs"][:, [0], :3, :, :]
            key_poses = key_poses.permute(1, 0, 2, 3, 4).reshape(-1, 3, 256, 256)
            key_poses = key_poses.permute(0, 2, 3, 1)
            key_poses = key_poses.detach().cpu().numpy()
            key_poses = np.clip(key_poses, 0, 1)
            key_poses = (key_poses * 255).astype(np.uint8)
            clip = ImageSequenceClip(list(key_poses), fps=1)
            clip.write_videofile(f"key_poses_{i}.mp4")
    # print(f"Processed {i} samples")
    print(f"Mean: {np.mean(lens)}")
    print(f"Max: {np.max(lens)}")
    print(f"Min: {np.min(lens)}")
    print(f"Std: {np.std(lens)}")
    print(f"Median: {np.median(lens)}")

    trajectories_xyz = torch.cat(trajectories_xyz, dim=0)
    print(f"Trajectory shape: {trajectories_xyz.shape}")
    print(f"Trajectory mean: {trajectories_xyz.mean(dim=0).mean(dim=0)}")
    print(f"Trajectory std: {trajectories_xyz.std(dim=0).mean(dim=0)}")
    print(f"Trajectory max: {trajectories_xyz.max(dim=0)[0].max(dim=0)[0]}")
    print(f"Trajectory min: {trajectories_xyz.min(dim=0)[0].min(dim=0)[0]}")
    print(f"Trajectory median: {trajectories_xyz.median(dim=0)[0].median(dim=0)[0]}")
