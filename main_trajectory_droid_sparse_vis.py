"""Main script for trajectory optimization."""
import os
import random
import numpy as np
import torch
import tap
from matplotlib import pyplot as plt
from pathlib import Path
import blosc
import pickle
import cv2
from typing import Tuple, Optional
import io
from engine import BaseTrainTester
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from utils.common_utils import (
    load_instructions, get_gripper_loc_bounds, count_parameters
)
from main_trajectory import (
    traj_collate_fn,
    TrainTester as BaseTrainTester,
    generate_visualizations
)
from torch.utils.tensorboard import SummaryWriter

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
    is_2d_model: int = 0

class TrainTester(BaseTrainTester):
    """Train/test a trajectory optimization algorithm."""
    def __init__(self, args):
        """Initialize."""

        self.args = args

        self.writer = SummaryWriter(log_dir=args.log_dir)


    def get_model(self):
        """Initialize the model."""
        # Initialize model with arguments

        if self.args.is_2d_model:
            from diffuser_actor.trajectory_optimization.diffuser_actor_2d import DiffuserActor
        else:
            from diffuser_actor.trajectory_optimization.diffuser_actor import DiffuserActor
        
        _model = DiffuserActor(
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
        print("Model parameters:", count_parameters(_model))

        return _model
    def get_datasets(self):
        """Initialize datasets."""
        # Initialize datasets with arguments
        if self.args.is_2d_model:
            from datasets.dataset_droid_sparse_2d import RLBenchDataset
        else:
            from datasets.dataset_droid_sparse import RLBenchDataset
        
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
    
    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, data, 
                        split='val'):
        """Run a given number of evaluation steps."""
        
        
        values = {}
        device = next(model.parameters()).device
        model.eval()

        for i, sample in enumerate([data]):
            
            
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
                sample["instr"].to(device),
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
            if dist.get_rank() == 0: # 
                generate_visualizations_with_pcd(
                    sample["pcds"].to(device),
                    sample["rgbs"].to(device),
                    curr_gripper.to(device),
                    action,
                    sample["trajectory"].to(device),
                    sample["trajectory_mask"].to(device),
                    is_2d_model=self.args.is_2d_model
                )
        values = {k: v.mean().item() for k, v in values.items()}
        for key, value in values.items():
            print(f"{key}: {value:.03f}")
        # Log all statistics
        # values = self.synchronize_between_processes(values)
        # values = {k: v.mean().item() for k, v in values.items()}
        # if dist.get_rank() == 0:
        #     if step_id > -1:
        #         for key, val in values.items():
        #             self.writer.add_scalar(key, val, step_id)

        #     # Also log to terminal
        #     print(f"Step {step_id}:")
        #     for key, value in values.items():
        #         print(f"{key}: {value:.03f}")

        # return values.get('val-losses/traj_pos_acc_001', None)


    def main(self, data):
        """Run visualization."""
        # Get loaders
        # train_loader, test_loader = self.get_loaders(collate_fn)

        # Get model
        # breakpoint()
        model = self.get_model()

        # Get criterion
        criterion = self.get_criterion()

        # Get optimizer
        optimizer = self.get_optimizer(model)

        # Move model to devices
        if torch.cuda.is_available():
            model = model.cuda()
        model = DistributedDataParallel(
            model, device_ids=[self.args.local_rank],
            broadcast_buffers=False, find_unused_parameters=True
        )

        # Check for a checkpoint
        start_iter, best_loss = 0, None
        if self.args.checkpoint:
            assert os.path.isfile(self.args.checkpoint)
            start_iter, best_loss = self.load_checkpoint(model, optimizer, use_ddp=True)

        # Eval only
        if bool(self.args.eval_only):
            print("Test evaluation.......")
            model.eval()
            new_loss = self.evaluate_nsteps(
                model, criterion, data, 
            )
            return model


def generate_visualizations_with_pcd(visible_pcd, visible_rgb, curr_gripper_history, preds, gts, masks, box_size=0.3, is_2d_model=False):
    """Visualize by plotting the point clouds and gripper pose.

    Args:
        visible_pcd: An array of shape (B, ncam, 3, H, W)
        visible_rgb: An array of shape (B, ncam, 3, H, W)
        curr_gripper: An array of shape (B, nhist, 8)
    """

    fig = plt.figure()
    canvas = fig.canvas
    images = []
    for batch_idx in range(visible_pcd.shape[0]):

        # plt.imshow(visible_rgb[batch_idx].permute(0, 2, 3, 1)[0].detach().cpu().numpy())
        # plt.show()
        images.append((visible_rgb[batch_idx].permute(0, 2, 3, 1)[0].detach().cpu().numpy()*255).astype(np.uint8)[..., ::-1]) # to bgr for cv2
        if not is_2d_model:
            # plt.imshow(visible_rgb[batch_idx].permute(0, 2, 3, 1)[1].detach().cpu().numpy())
            # plt.show()
            images.append((visible_rgb[batch_idx].permute(0, 2, 3, 1)[1].detach().cpu().numpy()*255).astype(np.uint8)[..., ::-1])

        cur_vis_pcd = visible_pcd[batch_idx].permute(0, 2, 3, 1).reshape(-1, 3).detach().cpu().numpy() # (ncam * H * W, 3)
        cur_vis_rgb = visible_rgb[batch_idx].permute(0, 2, 3, 1).reshape(-1, 3).detach().cpu().numpy()#[..., ::-1] # (ncam * H * W, 3)
        curr_gripper = curr_gripper_history[batch_idx, -1].detach().cpu().numpy()
        rand_inds = np.random.choice(cur_vis_pcd.shape[0], 20000, replace=False)
        mask_ = (
                (cur_vis_pcd[rand_inds, 2] >= -0.2) &
                (cur_vis_pcd[rand_inds, 2] <= 0.7) &
                (cur_vis_pcd[rand_inds, 1] >= -0.7) &
                (cur_vis_pcd[rand_inds, 1] <= 0.7) &
                (cur_vis_pcd[rand_inds, 0] >= -0.1) &
                (cur_vis_pcd[rand_inds, 0] <= 1.3)
            )
        rand_inds = rand_inds[mask_]
        
        pred = preds[batch_idx].detach().cpu().numpy()
        gt = gts[batch_idx].detach().cpu().numpy()
        mask = masks[batch_idx].detach().cpu().numpy()

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        ax.scatter3D(
            pred[~mask][:, 0], pred[~mask][:, 1], pred[~mask][:, 2],
            color='red', label='pred', s=90
        )
        ax.scatter3D(
            gt[~mask][:, 0], gt[~mask][:, 1], gt[~mask][:, 2],
            color='blue', label='gt', s=90
        )

        ax.scatter(cur_vis_pcd[rand_inds, 0],
                cur_vis_pcd[rand_inds, 1],
                cur_vis_pcd[rand_inds, 2],
                c=np.clip(cur_vis_rgb[rand_inds], 0, 1), s=12)
        # plot the origin
        ax.scatter(0, 0, 0, c='g', s=130)
        # plot the gripper pose
        ax.scatter(curr_gripper[0], curr_gripper[1], curr_gripper[2], c='y', s=200)

        for elev, azim in zip([20, 25, 25, 20, 35],
                          [90, 135, 225, 270, 360]):
            ax.view_init(elev=elev, azim=azim, roll=0)
            # ax.set_xlim(center[0] - box_size, center[0] + box_size)
            # ax.set_ylim(center[1] - box_size, center[1] + box_size)
            # ax.set_zlim(center[2] - box_size, center[2] + box_size)
            ax.set_ylim([-0.6, 0.6])
            ax.set_xlim([0, 1.1])
            ax.set_zlim([-0.2, 0.7])
            # add axes label
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])
            # ax.set_zticklabels([])
            plt.legend()
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            image = fig_to_numpy(fig, dpi=120)
            # # plt.show()
            # ax.set_title("Trajectory Visualization")
            # # add legend
            # ax.legend()
            image = image[340:-340, 340:-340] # HACK <>
            image = cv2.resize(image, (256, 256))
            images.append(image)
    if is_2d_model:
        images = np.concatenate([
            np.concatenate(images[k*6:(k+1)*6], axis=1) for k in range(len(images) // 6)
        ], axis=0)
    else:
        images = np.concatenate([
            np.concatenate(images[k*7:(k+1)*7], axis=1) for k in range(len(images) // 7)
        ], axis=0)
    # plt.imshow(images)
    # plt.show()
    cv2.imshow("Trajectory Visualization", images)
    #save
    cv2.imwrite("trajectory_visualization.png", images)
    # wait forever
    # cv2.waitKey(0)

def fig_to_numpy(fig, dpi=60):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    return img


def __getitem__(episode, use_wrist_camera=True, is_2d_model=False):
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
    

    # Get frame ids for this chunk
    frame_ids = episode[0][0:5]

    # Get the image tensors for the frame ids we got
    states = torch.stack([
        episode[1][i] if isinstance(episode[1][i], torch.Tensor)
        else torch.from_numpy(episode[1][i])
        for i in frame_ids
    ])


    # Split RGB and XYZ
    rgbs = states[:, :, 0]
    pcds = states[:, :, 1]
    # rgbs = self._unnormalize_rgb(rgbs)

    # Get action tensors for respective frame ids
    action = torch.cat([torch.from_numpy(episode[2][i]) for i in frame_ids])

    # Sample one instruction feature
    if True:
        # instr = random.choice(torch.from_numpy(episode[7]))
        instr = torch.from_numpy(episode[7])[0]
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
    from datasets.utils import TrajectoryInterpolator
    _interpolate_traj = TrajectoryInterpolator(
                use=True,
                interpolation_length=20
            )
    if True:
        if True:
            traj_items = [
                _interpolate_traj(torch.from_numpy(episode[5][i])) for i in frame_ids
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

    
    
    if is_2d_model or not use_wrist_camera:
        rgbs = rgbs[:, :-1]
        pcds = pcds[:, :-1]

    if is_2d_model:
        # rgbs = rgbs[:, :1]
        # pcds = pcds[:, :1]
        rgbs = rgbs[:, 1:2]
        pcds = pcds[:, 1:2]

    ret_dict = {
        "task": ["dummy" for _ in frame_ids],
        "rgbs": rgbs.float(),  # e.g. tensor (n_frames, n_cam, 3, H, W)
        "pcds": pcds.float(),  # e.g. tensor (n_frames, n_cam, 3, H, W)
        "action": action.float(),  # e.g. tensor (n_frames, 8), target pose
        "instr": instr.float(),  # a (n_frames, 53, 512) tensor
        "curr_gripper": gripper.float(), # e.g. tensor (n_frames, 8)
        "curr_gripper_history": gripper_history.float() # e.g. tensor (n_frames, 3, 8)
    }
    if True:
        ret_dict.update({
            "trajectory": traj.float(),  # e.g. tensor (n_frames, T, 8)
            "trajectory_mask": traj_mask.bool()  # tensor (n_frames, T)
        })
    print("languages", episode[6])
    return ret_dict

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

    # path = "/home/lawchen/project/droid/data/train/Fri_Apr__7_13:31:28_2023/sparse_trajectory.dat" # first 100
    # path = "/home/lawchen/project/droid/data/train/Fri_Apr_21_00:07:16_2023/sparse_trajectory.dat" # first 100
    # path = "/home/lawchen/project/droid/data/train/Fri_Nov__3_16:47:08_2023/sparse_trajectory.dat" # good <-- use
    # path = "/home/lawchen/project/droid/data/train/Mon_Sep_25_08:36:07_2023/sparse_trajectory.dat" # okay
    # path = "/home/lawchen/project/droid/data/train/Sat_Dec__2_14:04:31_2023/sparse_trajectory.dat" # okay
    # path = "/home/lawchen/project/droid/data/train/Thu_Nov_23_12:09:39_2023/sparse_trajectory.dat" # okay
    path = "/home/lawchen/project/droid/data/train/Thu_Oct_19_15:16:22_2023/sparse_trajectory.dat" # okay <-- use
    # path = "/home/lawchen/project/droid/data/train/Wed_Sep_20_18:09:39_2023/sparse_trajectory.dat" # okay

    path = "/home/lawchen/project/droid/data/test/Sat_Jul_15_11:57:20_2023/sparse_trajectory.dat" # good
    # path = "/home/lawchen/project/droid/data/test/Sun_Nov__5_10:48:57_2023/sparse_trajectory.dat"
    path = "/home/lawchen/project/droid/data/test/Wed_Oct__4_10:49:05_2023/sparse_trajectory.dat" # good

    def read_compressed_pickle(file_path):
        """
        Read and decompress a .dat file containing a pickled object compressed with blosc.
        
        Args:
            file_path (str): Path to the .dat file.
            
        Returns:
            object: The deserialized object from the file.
        """
        with open(file_path, 'rb') as f:
            # Read the compressed data
            compressed_data = f.read()
            
            # Decompress the data using blosc
            decompressed_data = blosc.decompress(compressed_data)
            
            # Deserialize the data using pickle
            data = pickle.loads(decompressed_data)
            
        return data

    data = read_compressed_pickle(path)
    data = __getitem__(data, use_wrist_camera=args.use_wrist_camera, is_2d_model=args.is_2d_model)
    

    # Run
    train_tester = TrainTester(args)
    # train_tester.main(traj_collate_fn)
    train_tester.main(data)