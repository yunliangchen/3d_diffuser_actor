from collections import defaultdict, Counter
import itertools
import math
import random
from pathlib import Path
from time import time
import os
import glob
import torch
from torch.utils.data import Dataset

from .utils import loader, Resize, TrajectoryInterpolator


class RLBenchDataset(Dataset):
    """RLBench dataset."""

    def __init__(
        self,
        # required
        root,
        # dataset specification
        max_episode_length=10,
        cache_size=0,
        num_iters=None,
        cameras=("wrist", "left_shoulder", "right_shoulder"),
        # for augmentations
        training=True,
        image_rescale=(1.0, 1.0),
        # for trajectories
        return_low_lvl_trajectory=False,
        dense_interpolation=False,
        interpolation_length=100,
        relative_action=False,
        use_wrist_camera=True
    ):
        self._cache = {}
        self._cache_size = cache_size
        self._cameras = cameras
        self._max_episode_length = max_episode_length
        self._num_iters = num_iters
        self._training = training
        self._return_low_lvl_trajectory = return_low_lvl_trajectory
        self._root = root
        self._relative_action = relative_action
        self._use_wrist_camera = use_wrist_camera

        # For trajectory optimization, initialize interpolation tools
        if return_low_lvl_trajectory:
            assert dense_interpolation
            self._interpolate_traj = TrajectoryInterpolator(
                use=dense_interpolation,
                interpolation_length=interpolation_length
            )

        # If training, initialize augmentation classes
        if self._training:
            self._resize = Resize(scales=image_rescale)

        # File-names of episodes per task and variation
        episodes_by_task = []  # [filepath]
        # list all subfolders in the root
        for root, _, files in os.walk(self._root):
            # Search for .dat files in the current directory
            for file in glob.glob(os.path.join(root, '*.dat')):
                episodes_by_task.append(file)

        # sort episodes_by_task and take the first 100
        episodes_by_task = sorted(episodes_by_task)[:100]
        
        # Collect and trim all episodes in the dataset
        self._episodes = episodes_by_task
        self._num_episodes = len(episodes_by_task)
        print(f"Created dataset from {root} with {self._num_episodes}")
        self._episodes_by_task = episodes_by_task

    def read_from_cache(self, args):
        if self._cache_size == 0:
            return loader(args)

        if args in self._cache:
            return self._cache[args]

        value = loader(args)

        if len(self._cache) == self._cache_size:
            key = list(self._cache.keys())[int(time()) % self._cache_size]
            del self._cache[key]

        if len(self._cache) < self._cache_size:
            self._cache[args] = value

        return value

    @staticmethod
    def _unnormalize_rgb(rgb):
        # (from [-1, 1] to [0, 1]) to feed RGB to pre-trained backbone
        return rgb / 2 + 0.5

    def __getitem__(self, episode_id):
        """
        the episode item: [
            [frame_ids],  # we use chunk and max_episode_length to index it
            [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256)
                obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
            [action_tensors],  # wrt frame_ids, (1, 8)
            [camera_dicts],
            [gripper_tensors],  # wrt frame_ids, (1, 8)
            [trajectories]  # wrt frame_ids, (N_i, 8)
            [languages],  # list of strings
            [languages_embedding_arrays]  # list of tensors
        ]
        """
        episode_id %= self._num_episodes
        file = self._episodes[episode_id]

        # Load episode
        episode = self.read_from_cache(file)
        if episode is None:
            return None

        # Dynamic chunking so as not to overload GPU memory
        chunk = random.randint(
            0, math.ceil(len(episode[0]) / self._max_episode_length) - 1
        )

        # Get frame ids for this chunk
        frame_ids = episode[0][
            chunk * self._max_episode_length:
            (chunk + 1) * self._max_episode_length
        ]

        # Get the image tensors for the frame ids we got
        states = torch.stack([
            episode[1][i] if isinstance(episode[1][i], torch.Tensor)
            else torch.from_numpy(episode[1][i])
            for i in frame_ids
        ])

        # Camera ids
        # if episode[3]:
        #     cameras = list(episode[3][0].keys())
        #     assert all(c in cameras for c in self._cameras)
        #     index = torch.tensor([cameras.index(c) for c in self._cameras])
        #     # Re-map states based on camera ids
        #     states = states[:, index]

        # Split RGB and XYZ
        rgbs = states[:, :, 0]
        pcds = states[:, :, 1]
        # rgbs = self._unnormalize_rgb(rgbs)

        # Get action tensors for respective frame ids
        action = torch.cat([torch.from_numpy(episode[2][i]) for i in frame_ids])

        # Sample one instruction feature
        if True:
            instr = random.choice(torch.from_numpy(episode[7]))
            instr = instr.repeat(len(rgbs), 1, 1)
        else:
            instr = torch.zeros((rgbs.shape[0], 53, 512))

        # Get gripper tensors for respective frame ids
        gripper = torch.stack([torch.from_numpy(episode[4][i]) for i in frame_ids])

        # gripper history
        gripper_history = torch.stack([
            torch.stack([torch.from_numpy(episode[4])[max(0, i-2)] for i in frame_ids]),
            torch.stack([torch.from_numpy(episode[4])[max(0, i-1)] for i in frame_ids]),
            gripper
        ], dim=1)

        # Low-level trajectory
        traj, traj_lens = None, 0
        if self._return_low_lvl_trajectory:
            if True:
                traj_items = [
                    self._interpolate_traj(torch.from_numpy(episode[5][i])) for i in frame_ids
                ]
            else:
                traj_items = [
                    self._interpolate_traj(
                        torch.cat([episode[4][i], episode[2][i]], dim=0)
                    ) for i in frame_ids
                ]
            max_l = max(len(item) for item in traj_items)
            traj = torch.zeros(len(traj_items), max_l, 8)
            traj_lens = torch.as_tensor(
                [len(item) for item in traj_items]
            )
            for i, item in enumerate(traj_items):
                traj[i, :len(item)] = item
            traj_mask = torch.zeros(traj.shape[:-1])
            for i, len_ in enumerate(traj_lens.long()):
                traj_mask[i, len_:] = 1

        # Augmentations
        if self._training:
            if traj is not None:
                for t, tlen in enumerate(traj_lens):
                    traj[t, tlen:] = 0
            modals = self._resize(rgbs=rgbs, pcds=pcds)
            rgbs = modals["rgbs"]
            pcds = modals["pcds"]
        
        if not self._use_wrist_camera:
            rgbs = rgbs[:, :-1]
            pcds = pcds[:, :-1]

        # randomly sample one of the remaining cameras
        num_cameras = rgbs.shape[1]
        if num_cameras > 1:
            cam_index = random.randint(0, num_cameras - 1)
            rgbs = rgbs[:, [cam_index]]
            pcds = pcds[:, [cam_index]]        

        ret_dict = {
            "task": ["dummy" for _ in frame_ids],
            "rgbs": rgbs,  # e.g. tensor (n_frames, n_cam, 3+1, H, W)
            "pcds": pcds,  # e.g. tensor (n_frames, n_cam, 3, H, W)
            "action": action,  # e.g. tensor (n_frames, 8), target pose
            "instr": instr,  # a (n_frames, 53, 512) tensor
            "curr_gripper": gripper,
            "curr_gripper_history": gripper_history
        }
        if self._return_low_lvl_trajectory:
            ret_dict.update({
                "trajectory": traj,  # e.g. tensor (n_frames, T, 8)
                "trajectory_mask": traj_mask.bool()  # tensor (n_frames, T)
            })

        # check there is no nan in the data
        # check = [rgbs, pcds, action, instr, gripper, gripper_history]
        # for i in range(len(check)):
        #     if torch.isnan(check[i]).any():
        #         breakpoint()
        #         print(f"nan in {i}")
        #         return None
        return ret_dict

    def __len__(self):
        if self._num_iters is not None:
            return self._num_iters
        return self._num_episodes