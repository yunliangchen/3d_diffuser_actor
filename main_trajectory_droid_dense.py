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
    TrainTester as BaseTrainTester,
    generate_visualizations
)
from datasets.dataset_droid_dense import DROIDDataset3dDA


def traj_collate_fn(batch):
    keys = [
        "trajectory", "trajectory_mask",
        "rgbs", "pcds",
        "curr_gripper", "curr_gripper_history", "action"
    ]
    ret_dict = {
        key: torch.cat([
            item[key].float() if key not in ['trajectory_mask', 'instr'] else item[key]
            for item in batch
        ]) for key in keys
    }

    ret_dict["task"] = []
    ret_dict["instr"] = [item['instr'] for item in batch]
    for item in batch:
        ret_dict["task"] += item['task']
    return ret_dict



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
    use_wrist_camera: int = 0

    # Logging to base_log_dir/exp_log_dir/run_log_dir
    base_log_dir: Path = Path(__file__).parent / "train_logs"
    exp_log_dir: str = "exp"
    run_log_dir: str = "run"

    # Main training parameters
    num_workers: int = 1
    batch_size: int = 16
    batch_size_val: int = 4
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
    fps_subsampling_factor: int = 5
    use_preenc_language: int = 0

class TrainTester(BaseTrainTester):
    """Train/test a trajectory optimization algorithm."""
    def __init__(self, args):
        """Initialize."""
        super().__init__(args)
    def get_datasets(self):
        """Initialize datasets."""
        # Initialize datasets with arguments
        train_dataset = DROIDDataset3dDA(root=self.args.dataset, use_wrist_camera=bool(self.args.use_wrist_camera))
        test_dataset = DROIDDataset3dDA(root=self.args.dataset, use_wrist_camera=bool(self.args.use_wrist_camera))
        # train_dataset = DROIDDataset(
        #     root=self.args.dataset,
        #     instructions=instruction,
        #     taskvar=taskvar,
        #     max_episode_length=self.args.max_episode_length,
        #     cache_size=self.args.cache_size,
        #     max_episodes_per_task=self.args.max_episodes_per_task,
        #     num_iters=self.args.train_iters,
        #     cameras=self.args.cameras,
        #     training=True,
        #     image_rescale=tuple(
        #         float(x) for x in self.args.image_rescale.split(",")
        #     ),
        #     return_low_lvl_trajectory=True,
        #     dense_interpolation=bool(self.args.dense_interpolation),
        #     interpolation_length=self.args.interpolation_length
        # )
        # test_dataset = DROIDDataset(
        #     root=self.args.valset,
        #     instructions=instruction,
        #     taskvar=taskvar,
        #     max_episode_length=self.args.max_episode_length,
        #     cache_size=self.args.cache_size_val,
        #     max_episodes_per_task=self.args.max_episodes_per_task,
        #     cameras=self.args.cameras,
        #     training=False,
        #     image_rescale=tuple(
        #         float(x) for x in self.args.image_rescale.split(",")
        #     ),
        #     return_low_lvl_trajectory=True,
        #     dense_interpolation=bool(self.args.dense_interpolation),
        #     interpolation_length=self.args.interpolation_length
        # )
        return train_dataset, test_dataset
    
    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, step_id, val_iters,
                        split='val'):
        """Run a given number of evaluation steps."""
        if self.args.val_iters != -1:
            val_iters = self.args.val_iters
        values = {}
        device = next(model.parameters()).device
        model.eval()

        for i, sample in enumerate(loader):
            if i == val_iters:
                break

            if self.args.keypose_only:
                sample["trajectory"] = sample["trajectory"][:, [-1]]
                sample["trajectory_mask"] = sample["trajectory_mask"][:, [-1]]
            else:
                sample["trajectory"] = sample["trajectory"][:, 1:]
                sample["trajectory_mask"] = sample["trajectory_mask"][:, 1:]

            curr_gripper = (
                sample["curr_gripper"] if self.args.num_history < 1
                else sample["curr_gripper_history"][:, -self.args.num_history:]
            )
            action = model(
                sample["trajectory"].to(device),
                sample["trajectory_mask"].to(device),
                sample["rgbs"].to(device),
                sample["pcds"].to(device),
                sample["instr"],
                curr_gripper.to(device),
                run_inference=True
            )
            losses, losses_B = criterion.compute_metrics(
                action,
                sample["trajectory"].to(device),
                sample["trajectory_mask"].to(device)
            )

            # Gather global statistics
            for n, l in losses.items():
                key = f"{split}-losses/mean/{n}"
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

            
            # Generate visualizations
            if i == 0 and dist.get_rank() == 0 and step_id > -1:
                viz_key = f'{split}-viz/viz'
                viz = generate_visualizations(
                    action,
                    sample["trajectory"].to(device),
                    sample["trajectory_mask"].to(device)
                )
                self.writer.add_image(viz_key, viz, step_id)

        # Log all statistics
        values = self.synchronize_between_processes(values)
        values = {k: v.mean().item() for k, v in values.items()}
        if dist.get_rank() == 0:
            if step_id > -1:
                for key, val in values.items():
                    self.writer.add_scalar(key, val, step_id)

            # Also log to terminal
            print(f"Step {step_id}:")
            for key, value in values.items():
                print(f"{key}: {value:.03f}")

        return values.get('val-losses/traj_pos_acc_001', None)


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Arguments
    args = Arguments().parse_args()
    print("Arguments:")
    print(args)
    print("-" * 100)
    if args.gripper_loc_bounds is None:
        args.gripper_loc_bounds = np.array([[ 0.26358891, -0.13902004,  0.20537288], [0.74897486, 0.26469305, 0.52306706]]) * 1.0
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