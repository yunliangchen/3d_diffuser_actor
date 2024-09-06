import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
from scipy.spatial.transform import Rotation as R
import blosc
import torch
import PIL.Image as Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip



# path = "/lustre/fsw/portfolios/nvr/users/lawchen/droid_raw/droid_raw/droid_with_pcd_20k/Thu_Aug_10_15:56:51_2023/sparse_trajectory.dat"
path = "/lustre/fsw/portfolios/nvr/users/lawchen/droid_raw/droid_raw/droid_with_pcd_20k/Thu_Nov__2_21:23:36_2023/sparse_trajectory.dat"
path = "/lustre/fsw/portfolios/nvr/users/lawchen/droid_raw/droid_raw/droid_with_pcd_20k/Sat_Oct_14_11:35:33_2023/sparse_trajectory.dat"
path = "/lustre/fsw/portfolios/nvr/users/lawchen/droid_raw/droid_raw/droid_with_pcd_20k/Tue_May_30_17:53:59_2023/sparse_trajectory.dat"
path = "/lustre/fsw/portfolios/nvr/users/lawchen/droid_raw/droid_raw/droid_with_pcd_5k/Tue_Oct_17_23:02:16_2023/sparse_trajectory.dat"

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

frame_ids, obs_arrays, action_tensors, camera_dicts, gripper_tensors, trajectories, languages, languages_embedding_arrays = data

obs_arrays = np.transpose(obs_arrays, (0, 1, 2, 4, 5, 3)) # (n_frames, n_cam, [rgb, pcd], 3, H, W) -> (n_frames, n_cam, [rgb, pcd], H, W, 3)


def visualize_actions_and_point_clouds(visible_pcd, visible_rgb, cartesian_position_state,
                                       rand_inds=None, seg_mask=None):
    """Visualize by plotting the point clouds and gripper pose.

    Args:
        visible_pcd: An array of shape (ncam, H, W, 3)
        visible_rgb: An array of shape (ncam, H, W, 3)
        cartesian_position_state: A 3D array of the cartesian position of the gripper.
    """

    cur_vis_pcd = np.transpose(visible_pcd, (0, 1, 2, 3)).reshape(-1, 3) # (ncam * H * W, 3)
    cur_vis_rgb = np.transpose(visible_rgb, (0, 1, 2, 3)).reshape(-1, 3)#[..., ::-1] # (ncam * H * W, 3)
    if rand_inds is None:
        rand_inds = np.random.choice(cur_vis_pcd.shape[0], 20000, replace=False)
        mask = (
                (cur_vis_pcd[rand_inds, 2] >= -0.1) &
                (cur_vis_pcd[rand_inds, 2] <= 0.7) &
                (cur_vis_pcd[rand_inds, 1] >= -1) &
                (cur_vis_pcd[rand_inds, 1] <= 1) &
                (cur_vis_pcd[rand_inds, 0] >= -0.1) &
                (cur_vis_pcd[rand_inds, 0] <= 1.3)
            )
        rand_inds = rand_inds[mask]
        # if seg_mask is not None:
        #     mask = seg_mask[0].flatten()[rand_inds] > 1
        # else:
        #     mask = (
        #         (cur_vis_pcd[rand_inds, 2] >= -0.0) &
        #         # (cur_vis_pcd[rand_inds, 1] >= -1) &
        #         (cur_vis_pcd[rand_inds, 1] <= 5) &
        #         # (cur_vis_pcd[rand_inds, 0] >= -1) &
        #         (cur_vis_pcd[rand_inds, 0] <= 5)
        #     )
        # rand_inds = rand_inds[mask]
    fig = plt.figure()
    canvas = fig.canvas
    ax = fig.add_subplot(projection='3d')
    # ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    # breakpoint()
    ax.scatter(cur_vis_pcd[rand_inds, 0],
               cur_vis_pcd[rand_inds, 1],
               cur_vis_pcd[rand_inds, 2],
               c=np.clip(cur_vis_rgb[rand_inds].astype(float) *255/ 255, 0, 1), s=15)
    # mask = seg_mask[0].flatten()[rand_inds] > 1
    # ax.scatter(cur_vis_pcd[rand_inds[mask], 0],
    #            cur_vis_pcd[rand_inds[mask], 1],
    #            cur_vis_pcd[rand_inds[mask], 2],
    #            c=cur_vis_rgb[rand_inds[mask]], s=1)

    # plot the gripper pose
    ax.scatter(cartesian_position_state[0], cartesian_position_state[1], cartesian_position_state[2], c='b', s=100)
    # plot the origin
    ax.scatter(0, 0, 0, c='g', s=100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_ylim([-0.9, 0.9])
    ax.set_xlim([0, 1.5])
    ax.set_zlim([-0.2, 0.7])

    fig.tight_layout()
    # make an interactive 3d plot
    # plt.show()

    images = []
    for elev, azim in zip([10, 15, 20, 25, 30, 25, 20, 15, 45, 90],
                          [0, 45, 90, 135, 180, 225, 270, 315, 360, 360]):
    # for elev, azim in zip([10], [0]):
        ax.view_init(elev=elev, azim=azim, roll=0)
        ax.set_ylim([-0.9, 0.9])
        ax.set_xlim([0, 1.5])
        ax.set_zlim([-0.2, 0.7])
        # add axes label
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        canvas.draw()
        image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)
        image = image[60:, 110:-110] # HACK <>
        image = cv2.resize(image, dsize=None, fx=0.75, fy=0.75)
        images.append(image)
    images = np.concatenate([
        np.concatenate(images[:5], axis=1),
        np.concatenate(images[5:10], axis=1)
    ], axis=0)
    
    plt.close(fig)
    return images, rand_inds


def visualize_actions_and_point_clouds_video(visible_pcd, visible_rgb,
                                             gt_pose, 
                                             save=True, rotation_param="quat_from_query", cnt=None):
    """Visualize by plotting the point clouds and gripper pose as video.

    Args:
        visible_pcd: A tensor of shape (B, ncam, H, W, 3)
        visible_rgb: A tensor of shape (B, ncam, H, W, 3)
        gt_pose: A tensor of shape (B, 8)
        curr_pose: A tensor of shape (B, 8)
    """
    images, rand_inds = [], None
    for i in range(visible_pcd.shape[0]):
        image, rand_inds = visualize_actions_and_point_clouds(
            visible_pcd[i], visible_rgb[i],
            gt_pose[i], 
            rand_inds=rand_inds,
        )
        # add denoising progress bar
        images.append(image)
    # pil_images = []
    # for img in images:
    #     pil_images.extend([Image.fromarray(img)] * 2)
    # if cnt is None:
    #     pil_images[0].save("keypose_frames.gif", save_all=True,
    #                         append_images=pil_images[1:], duration=1, loop=0)
    # else:
    #     pil_images[0].save(f"keypose_frames{cnt}.gif", save_all=True,
    #                        append_images=pil_images[1:], duration=1, loop=0)
    clip = ImageSequenceClip(images, fps=1)
    if cnt is None:
        clip.write_videofile("keypose_frames.mp4")
    else:
        clip.write_videofile(f"keypose_frames{cnt}.mp4")


if __name__ == '__main__':
    all_images_all_timesteps = []
    all_points_all_timesteps = []
    for timestep in range(len(obs_arrays)):
        print("Timestep", timestep)

        cartesian_position_state_timestep = gripper_tensors[timestep]


        images_left = {"wrist": obs_arrays[timestep][2][0], "ext1": obs_arrays[timestep][0][0], "ext2": obs_arrays[timestep][1][0]}
        point_clouds_left = {"wrist": obs_arrays[timestep][2][1], "ext1": obs_arrays[timestep][0][1], "ext2": obs_arrays[timestep][1][1]}

        all_points_zed = []
        all_points_pcd = []
        all_images = []
        for camera_name in ["ext1", "ext2", "wrist"]:
            # left_image, right_image, depth_image, point_cloud = images_left[camera_name], images_right[camera_name], depth_left[camera_name], point_clouds_left[camera_name]
            left_image, point_cloud = images_left[camera_name], point_clouds_left[camera_name]
            
            # cv2.imshow(f"{camera_name} left", np.array(left_image*255).astype(np.uint8))
            # cv2.waitKey(1)
            cv2.imwrite(f"{camera_name}_left_{timestep}.png", cv2.cvtColor(np.array(left_image*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

            # import time
            # time.sleep(10)
            # plt.imshow(depth_image, cmap='jet')
            # plt.show()

            
            all_points_pcd.append(point_cloud)
            all_images.append(left_image)
        # all_points_zed = np.array(all_points_zed)
        all_points_pcd = np.array(all_points_pcd)
        all_images = np.array(all_images)
        # breakpoint()
        images_pcd, rand_inds = visualize_actions_and_point_clouds(all_points_pcd, all_images, cartesian_position_state_timestep[:3])
        # cv2.imshow("all images pcd", images_pcd)
        # cv2.waitKey(1)
        cv2.imwrite(f"all_images_pcd_{timestep}.png", cv2.cvtColor(images_pcd, cv2.COLOR_RGB2BGR))
        
        all_images_all_timesteps.append(all_images)
        all_points_all_timesteps.append(all_points_pcd)
    all_images_all_timesteps = np.array(all_images_all_timesteps) # (n_frames, n_cam, H, W, 3)
    all_points_all_timesteps = np.array(all_points_all_timesteps)

    
    
    
    # debugging visualization code
    visualize_actions_and_point_clouds_video(
        all_points_all_timesteps,
        all_images_all_timesteps,
        gripper_tensors[:3],
        
    )
        

            
