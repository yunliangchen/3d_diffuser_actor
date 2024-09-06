main_dir=droid_sparse_2d_distillation

dataset=/lustre/fsw/portfolios/nvr/users/lawchen/droid_raw/droid_raw/droid_with_pcd_20k
valset=/lustre/fsw/portfolios/nvr/users/lawchen/droid_raw/droid_raw/droid_with_pcd_5k

lr=1e-4 #3e-4
wd=5e-3
dense_interpolation=1
interpolation_length=20
use_wrist_camera=1
num_history=2
diffusion_timesteps=50
B=10 # per gpu batch size = B * num_frame per data loader. Actual batch size = B * num_gpu
accumulate_grad_batches=2
C=192 # embedding dim
ngpus=8
backbone=clip
image_size="256,256"
relative_action=0
fps_subsampling_factor=3
lang_enhanced=1
gripper_buffer=0.01
val_freq=300
quaternion_format=xyzw

run_log_dir=diffusion_taskABC_D-C$C-B$B-lr$lr-DI$dense_interpolation-$interpolation_length-H$num_history-DT$diffusion_timesteps-backbone$backbone-S$image_size-R$relative_action-wd$wd-usewrist$use_wrist_camera

export PYTHONPATH=`pwd`:$PYTHONPATH
# TODO:
# tune fps_subsampling_factor by visualizing applying directly on the point cloud and after noise point filtering
# tune diffusion_timesteps later (increase if needed)
# make embedding_dim C larger if needed

# fix image
# visualize point cloud
# feed in normalization bound - Done
# cap the max frames - Done (10)
# an improved 2d policy (RLBench for camera changes and naive 2d policy) but weak sell for project unless good argment, droid more robust
# vision encoder => 3d diffuser actor point cloud merging and convert to voxelization 
# unfreeze clipvision encoder + action


while true
do
    CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node 1 --master_port $RANDOM \
        main_trajectory_droid_sparse_2d_distillation.py \
        --backbone $backbone \
        --dataset $dataset \
        --valset $valset \
        --gripper_loc_bounds_buffer $gripper_buffer \
        --image_size $image_size \
        --num_workers 0 \
        --train_iters 600000 \
        --embedding_dim $C \
        --use_instruction 1 \
        --rotation_parametrization 6D \
        --diffusion_timesteps $diffusion_timesteps \
        --val_freq $val_freq \
        --val_iters 2 \
        --dense_interpolation $dense_interpolation \
        --interpolation_length $interpolation_length \
        --use_wrist_camera $use_wrist_camera \
        --exp_log_dir $main_dir \
        --batch_size 5 \
        --accumulate_grad_batches $accumulate_grad_batches \
        --batch_size_val 1 \
        --cache_size 50 \
        --cache_size_val 0 \
        --lr $lr\
        --wd $wd \
        --num_history $num_history \
        --relative_action $relative_action \
        --fps_subsampling_factor $fps_subsampling_factor \
        --lang_enhanced $lang_enhanced \
        --quaternion_format $quaternion_format \
        --run_log_dir $run_log_dir \
        --checkpoint_teacher /lustre/fsw/portfolios/nvr/users/lawchen/project/3d_diffuser_actor/train_logs/droid_sparse/diffusion_taskABC_D-C192-B15-lr1e-4-DI1-20-H2-DT50-backboneclip-S256,256-R0-wd5e-3-usewrist1/last.pth \
        --checkpoint /lustre/fsw/portfolios/nvr/users/lawchen/project/3d_diffuser_actor/train_logs/droid_sparse_2d_distillation/diffusion_taskABC_D-C192-B10-lr1e-4-DI1-20-H2-DT50-backboneclip-S256,256-R0-wd5e-3-usewrist1/last.pth
    sleep 1m
done
