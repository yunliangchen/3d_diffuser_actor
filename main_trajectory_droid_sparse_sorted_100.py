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
from utils.common_utils import (
    load_instructions, get_gripper_loc_bounds
)
from main_trajectory import (
    traj_collate_fn,
    TrainTester as BaseTrainTester,
    generate_visualizations
)
from datasets.dataset_droid_sparse_sorted_100 import RLBenchDataset


class Arguments(tap.Tap):
    image_size: str = "256,256"
    seed: int = 0
    checkpoint: Optional[Path] = None
    accumulate_grad_batches: int = 1
    val_freq: int = 500
    gripper_loc_bounds: Optional[str] = None
    gripper_loc_bounds_buffer: float = 0.04
    eval_only: int = 0

    # Training and validation datasets
    dataset: Path = None
    valset: Path = None
    dense_interpolation: int = 0
    interpolation_length: int = 100
    use_wrist_camera: int = 1

    # Logging to base_log_dir/exp_log_dir/run_log_dir
    base_log_dir: Path = Path(__file__).parent / "train_logs"
    exp_log_dir: str = "exp"
    run_log_dir: str = "run"

    # Main training parameters
    num_workers: int = 1
    batch_size: int = 16
    batch_size_val: int = 4
    cache_size: int = 100
    cache_size_val: int = 100
    lr: float = 1e-4
    wd: float = 5e-3  # used only for CALVIN
    train_iters: int = 200_000
    val_iters: int = -1  # -1 means heuristically-defined

    # Data augmentations
    image_rescale: str = "0.75,1.25"  # (min, max), "1.0,1.0" for no rescaling

    # Model
    backbone: str = "clip"  # one of "resnet", "clip"
    embedding_dim: int = 120
    num_vis_ins_attn_layers: int = 2
    use_instruction: int = 0
    rotation_parametrization: str = 'quat'
    quaternion_format: str = 'xyzw'
    diffusion_timesteps: int = 100
    keypose_only: int = 0
    num_history: int = 0
    relative_action: int = 0
    lang_enhanced: int = 0
    fps_subsampling_factor: int = 5,
    use_preenc_language: int = 1

class TrainTester(BaseTrainTester):
    """Train/test a trajectory optimization algorithm."""
    def __init__(self, args):
        """Initialize."""
        super().__init__(args)
    def get_datasets(self):
        """Initialize datasets."""
        # Initialize datasets with arguments
        train_dataset = RLBenchDataset(
            root=self.args.dataset,
            cache_size=self.args.cache_size,
            num_iters=self.args.train_iters,
            training=True,
            image_rescale=tuple(
                float(x) for x in self.args.image_rescale.split(",")
            ),
            return_low_lvl_trajectory=True,
            dense_interpolation=bool(self.args.dense_interpolation),
            interpolation_length=self.args.interpolation_length,
            use_wrist_camera=bool(self.args.use_wrist_camera)
        )
        test_dataset = RLBenchDataset(
            root=self.args.valset,
            cache_size=self.args.cache_size_val,
            training=False,
            image_rescale=tuple(
                float(x) for x in self.args.image_rescale.split(",")
            ),
            return_low_lvl_trajectory=True,
            dense_interpolation=bool(self.args.dense_interpolation),
            interpolation_length=self.args.interpolation_length,
            use_wrist_camera=bool(self.args.use_wrist_camera)
        )
        return train_dataset, test_dataset
    
if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Arguments
    args = Arguments().parse_args()
    print("Arguments:")
    print(args)
    print("-" * 100)
    if args.gripper_loc_bounds is None:
        """
        Trajectory mean: tensor([0.5381, 0.0098, 0.2824])
        Trajectory std: tensor([0.1050, 0.1574, 0.1566])
        Trajectory median: tensor([0.5421, 0.0064, 0.2638])
        Trajectory max: tensor([0.8242, 0.6763, 0.9966])
        Trajectory min: tensor([ 0.0509, -0.6323, -0.1967])
        """
        args.gripper_loc_bounds = np.array([[ 0., -0.72,  -0.22], [0.9, 0.72, 1.02]]) * 1.0
    else:
        args.gripper_loc_bounds = get_gripper_loc_bounds(
            args.gripper_loc_bounds,
            task=args.tasks[0] if len(args.tasks) == 1 else None,
            buffer=args.gripper_loc_bounds_buffer,
        )
    log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Logging:", log_dir)
    print(
        "Available devices (CUDA_VISIBLE_DEVICES):",
        os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    print("Device count", torch.cuda.device_count())
    args.local_rank = int(os.environ["LOCAL_RANK"])
    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # DDP initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # Run
    train_tester = TrainTester(args)
    train_tester.main(collate_fn=traj_collate_fn)