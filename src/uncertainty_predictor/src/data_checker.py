import numpy as np
import os
from tqdm import tqdm
import shutil
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def main():
    # Load the data from the .npz file
    data_folder = os.path.join(os.path.dirname(__file__), '..', 'data_gps/dataset2')
    output_folder = os.path.join(os.path.dirname(__file__), '..', 'valid_data/data_gps/dataset2')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(data_folder)
    data_files = sorted([file for file in files if file.endswith('.npz')], reverse=False) # time increasing order
    prev_est_odom = np.array([0,0,0,0])
    previous_wrong_data = False
    removed_files = 0

    delta_x_list = []  # List to store delta_x values
    est_err_list =[]

    prev_data = np.load(os.path.join(data_folder, data_files[0]))
    for data_file in tqdm(data_files, desc="check data integrity"):
        full_path = os.path.join(data_folder, data_file)
        data = np.load(full_path)
        pointcloud = data['pointcloud']
        gt_odom = data['gt_odom']
        est_odom = data['est_odom']
        # gt_rotation = R.from_quat(gt_odom[3:])
        # gt_yaw = gt_rotation.as_euler('zyx')[0]
        # est_rotation = R.from_quat(est_odom[3:])
        # est_yaw = est_rotation.as_euler('zyx')[0]

        cur_gt_odom  = gt_odom
        cur_est_odom = est_odom

        condition1 = (np.abs(cur_est_odom[0]) < 0.01) and (not (np.abs(prev_est_odom[0]) < 0.01) or previous_wrong_data)
        condition2 = (np.abs(cur_est_odom[1]) < 0.01) and (not (np.abs(prev_est_odom[1]) < 0.01) or previous_wrong_data)
        condition3 = (np.abs(cur_est_odom[2]) < 0.01) and (not (np.abs(prev_est_odom[2]) < 0.01) or previous_wrong_data)

        if (condition1 and condition2) or condition3:
            previous_wrong_data = True
            removed_files += 1
        else:
            previous_wrong_data = False
            output_file_path = os.path.join(output_folder, data_file)
            np.savez(output_file_path, pointcloud=pointcloud, gt_odom=cur_gt_odom, est_odom=cur_est_odom)
            delta_x = (gt_odom[:3] - est_odom[:3]) - (prev_data['gt_odom'][:3] - prev_data['est_odom'][:3])
            delta_x_list.append(delta_x)
            est_err_list.append((gt_odom[:3] - est_odom[:3]))
            prev_data = data

        prev_est_odom=cur_est_odom
    
    total_files = len(data_files)
    remaining_files = total_files - removed_files
    print(f"Check result: Total {total_files} files, {removed_files} removed, {remaining_files} remain")

    # Convert delta_x_list to numpy array
    delta_x_array = np.array(delta_x_list)
    est_err_array =np.array(est_err_list)

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    for i in range(3):
        axs[i].plot(delta_x_array[:, i])
        axs[i].set_title(f'delta_x[{i}]')
        axs[i].set_xlabel('File Index')
        axs[i].set_ylabel('Value')

    fig2, axs2 = plt.subplots(3, 1, figsize=(8, 6))
    for i in range(3):
        axs2[i].plot(est_err_array[:, i])
        axs2[i].set_title(f'est_err[{i}]')
        axs2[i].set_xlabel('File Index')
        axs2[i].set_ylabel('Value')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
