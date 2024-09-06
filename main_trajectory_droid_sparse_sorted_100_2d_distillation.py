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
from torch.nn.parallel import DataParallel, DistributedDataParallel
from utils.common_utils import (
    load_instructions, count_parameters, get_gripper_loc_bounds
)
from main_trajectory import (
    traj_collate_fn,
    TrainTester as BaseTrainTester,
    generate_visualizations
)
from torch.utils.data import default_collate
from tqdm import trange
from datasets.dataset_droid_sparse_sorted_100 import RLBenchDataset
from diffuser_actor.trajectory_optimization.diffuser_actor_2d import DiffuserActor as DiffuserActor2D
from diffuser_actor.trajectory_optimization.diffuser_actor import DiffuserActor as DiffuserActor3D

class Arguments(tap.Tap):
    image_size: str = "256,256"
    seed: int = 0
    checkpoint: Optional[Path] = None
    checkpoint_teacher: Path = None
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

    def get_model(self):
        """Initialize the model."""
        # Initialize model with arguments
        _model = DiffuserActor2D(
            backbone=self.args.backbone,
            image_size=tuple(int(x) for x in self.args.image_size.split(",")),
            embedding_dim=self.args.embedding_dim,
            num_vis_ins_attn_layers=self.args.num_vis_ins_attn_layers,
            use_instruction=bool(self.args.use_instruction),
            fps_subsampling_factor=self.args.fps_subsampling_factor,
            gripper_loc_bounds=self.args.gripper_loc_bounds,
            rotation_parametrization=self.args.rotation_parametrization,
            quaternion_format=self.args.quaternion_format,
            diffusion_timesteps=self.args.diffusion_timesteps,
            nhist=self.args.num_history,
            relative=bool(self.args.relative_action),
            lang_enhanced=bool(self.args.lang_enhanced),
            use_preenc_language=bool(self.args.use_preenc_language),
        )
        print("Student model parameters:", count_parameters(_model))

        return _model
    
    def get_teacher_model(self):
        """Initialize the model."""
        # Initialize model with arguments
        _model = DiffuserActor3D(
            backbone=self.args.backbone,
            image_size=tuple(int(x) for x in self.args.image_size.split(",")),
            embedding_dim=self.args.embedding_dim,
            num_vis_ins_attn_layers=self.args.num_vis_ins_attn_layers,
            use_instruction=bool(self.args.use_instruction),
            fps_subsampling_factor=self.args.fps_subsampling_factor,
            gripper_loc_bounds=self.args.gripper_loc_bounds,
            rotation_parametrization=self.args.rotation_parametrization,
            quaternion_format=self.args.quaternion_format,
            diffusion_timesteps=self.args.diffusion_timesteps,
            nhist=self.args.num_history,
            relative=bool(self.args.relative_action),
            lang_enhanced=bool(self.args.lang_enhanced),
            use_preenc_language=bool(self.args.use_preenc_language),
        )
        print("Teacher model parameters:", count_parameters(_model))

        # set teacher model to eval mode
        _model.eval()
        # freeze teacher model
        for param in _model.parameters():
            param.requires_grad = False

        return _model

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
    
    def main(self, collate_fn=default_collate):
        """Run main training/testing pipeline."""
        # Get loaders
        train_loader, test_loader = self.get_loaders(collate_fn)

        # Get student model
        model = self.get_model()

        # Get teacher model
        teacher_model = self.get_teacher_model()
        
        # Get criterion
        criterion = self.get_criterion() # TODO: add distillation loss

        # Get optimizer
        optimizer = self.get_optimizer(model)

        # Move model to devices
        if torch.cuda.is_available():
            model = model.cuda()
            teacher_model = teacher_model.cuda()
        model = DistributedDataParallel(
            model, device_ids=[self.args.local_rank],
            broadcast_buffers=False, find_unused_parameters=True
        )
        teacher_model = DataParallel(
            teacher_model, device_ids=[self.args.local_rank]
        )

        # Check for a checkpoint
        start_iter, best_loss = 0, None
        if self.args.checkpoint:
            assert os.path.isfile(self.args.checkpoint)
            start_iter, best_loss = self.load_checkpoint(model, optimizer)
        if self.args.checkpoint_teacher:
            assert os.path.isfile(self.args.checkpoint_teacher)
            teacher_model = self.load_checkpoint_teacher(teacher_model)

        # Eval only
        if bool(self.args.eval_only):
            print("Test evaluation.......")
            model.eval()
            new_loss = self.evaluate_nsteps(
                model, criterion, test_loader, step_id=-1,
                val_iters=max(
                    5,
                    int(4 * 20/self.args.batch_size_val)
                )
            )
            return model

        # Training loop
        iter_loader = iter(train_loader)
        model.train()
        for step_id in trange(start_iter, self.args.train_iters):
            try:
                sample = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                sample = next(iter_loader)

            self.train_one_step(model, teacher_model, criterion, optimizer, step_id, sample)
            if (step_id + 1) % self.args.val_freq == 0:
                print("Train evaluation.......")
                model.eval()
                new_loss = self.evaluate_nsteps(
                    model, criterion, train_loader, step_id,
                    val_iters=max(
                        5,
                        int(4 * 20/self.args.batch_size_val)
                    ),
                    split='train'
                )
                print("Test evaluation.......")
                model.eval()
                new_loss = self.evaluate_nsteps(
                    model, criterion, test_loader, step_id,
                    val_iters=max(
                        5,
                        int(4 * 20/self.args.batch_size_val)
                    )
                )
                if dist.get_rank() == 0:  # save model
                    best_loss = self.save_checkpoint(
                        model, optimizer, step_id,
                        new_loss, best_loss
                    )
                model.train()

        return model
    
    def load_checkpoint_teacher(self, model):
        """Load from teacher's checkpoint."""
        print("=> loading checkpoint for the teacher '{}'".format(self.args.checkpoint_teacher))

        model_dict = torch.load(self.args.checkpoint_teacher, map_location="cpu")
        model.load_state_dict(model_dict["weight"])

        print("=> loaded successfully '{}' (step {})".format(
            self.args.checkpoint_teacher, model_dict.get("iter", 0)
        ))
        del model_dict
        torch.cuda.empty_cache()
        return model
    
    def train_one_step(self, model, teacher_model, criterion, optimizer, step_id, sample):
        """Run a single training step."""
        if step_id % self.args.accumulate_grad_batches == 0:
            optimizer.zero_grad()

        if self.args.keypose_only:
            sample["trajectory"] = sample["trajectory"][:, [-1]]
            sample["trajectory_mask"] = sample["trajectory_mask"][:, [-1]]
        else:
            sample["trajectory"] = sample["trajectory"][:, 1:]
            sample["trajectory_mask"] = sample["trajectory_mask"][:, 1:]

        # Forward pass
        curr_gripper = (
            sample["curr_gripper"] if self.args.num_history < 1
            else sample["curr_gripper_history"][:, -self.args.num_history:]
        )

        with torch.no_grad():
            _, sampled_inds, noise, timesteps, pred, gripper_features_all_layers, features_all_layers = teacher_model(
                sample["trajectory"],
                sample["trajectory_mask"],
                sample["rgbs"],
                sample["pcds"],
                sample["instr"],
                curr_gripper,
                distillation_mode=True
            )

        # randomly sample one of the cameras that are not wrist camera
        num_cameras = sample["rgbs"].shape[1]
        if self.args.use_wrist_camera: # the teacher policy uses wrist camera
            num_cameras -= 1
        cam_index = random.randint(0, num_cameras - 1)
        sample["rgbs"] = sample["rgbs"][:, [cam_index]]
        sample["pcds"] = sample["pcds"][:, [cam_index]]

        distillation_related_inputs = {
            "sampled_inds": sampled_inds,
            "noise": noise,
            "timesteps": timesteps,
            "pred": pred,
            "gripper_features_all_layers": gripper_features_all_layers,
            "features_all_layers": features_all_layers,
            "cam_index_for_2d_policy": cam_index
        }

        total_loss, detailed_loss = model(
            sample["trajectory"],
            sample["trajectory_mask"],
            sample["rgbs"],
            sample["pcds"],
            sample["instr"],
            curr_gripper,
            distillation_mode=True,
            distillation_related_inputs=distillation_related_inputs
        )

        # Backward pass
        loss = criterion.compute_loss(total_loss)
        loss.backward()

        # Update
        if step_id % self.args.accumulate_grad_batches == self.args.accumulate_grad_batches - 1:
            optimizer.step()

        # Log
        if dist.get_rank() == 0 and (step_id + 1) % self.args.val_freq == 0:
            self.writer.add_scalar("lr", self.args.lr, step_id)
            self.writer.add_scalar("train-loss/noise_mse", detailed_loss["target_loss"], step_id)
            self.writer.add_scalar("train-loss/pred_distillation_loss", detailed_loss["pred_l2_loss"], step_id)
            self.writer.add_scalar("train-loss/gripper_features_distillation_loss_crossattn", detailed_loss["gripper_features_l2_loss_part1"], step_id)
            self.writer.add_scalar("train-loss/gripper_features_distillation_loss_selfattn", detailed_loss["gripper_features_l2_loss_part2"], step_id)
            self.writer.add_scalar("train-loss/context_feature_distillation_loss", detailed_loss["context_feature_loss"], step_id)
            
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