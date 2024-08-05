import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from voxelmap import VoxelMap
from tqdm import tqdm
import utils
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Map:
    def __init__(self,data_folder=None,cfg_folder=None,voxel_size=None,min_xyz=None,max_xyz=None):
        if voxel_size==None and min_xyz==None and max_xyz==None:
            self.data_folder = data_folder
            cfg = utils.load_config(cfg_folder+'/'+'pointcloud_map.yaml')
            self.device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
            self.pointcloud_sampling = cfg['pointcloud_sampling_for_mapping']
            self.align_matrix = utils.define_align_matrix(cfg['base2camera'])
            self.align_matrix = torch.from_numpy(self.align_matrix).to(self.device).float()
            self.min_num_points = cfg['min_points_to_save']
            self.voxel_size = cfg['voxel_size']
            self.voxel_map=None
            self.min_xyz=None
            self.max_xyz=None
            self.map_size=None
            self.voxel_map, self.gt_odom_list, self.est_odom_list = self.generate_map()
        else:
            self.voxel_size=voxel_size
            self.min_xyz=min_xyz
            self.max_xyz=max_xyz
            self.map_size = np.ceil((self.max_xyz - self.min_xyz) / self.voxel_size).astype(int)
            self.voxel_map=np.empty(self.map_size, dtype=dict)
            for index in np.ndindex(tuple(self.map_size)):
                self.voxel_map[index] = {}
        
        
    def sample_pointcloud(self,pointcloud_in):
        sampled_index = torch.randperm(len(pointcloud_in), device=self.device)[:int(len(pointcloud_in)*self.pointcloud_sampling)].tolist()
        pointcloud_out = pointcloud_in[sampled_index].T # 6 x N
        return pointcloud_out
        
    def pointcloud_transform_local2world(self,local_pointcloud, gt_odom):
        x, y, z, qx, qy, qz, qw = gt_odom[0],gt_odom[1],gt_odom[2],gt_odom[3],gt_odom[4],gt_odom[5],gt_odom[6]
        q = np.array([qx, qy, qz, qw])
        q_norm = q / np.linalg.norm(q)
        r = R.from_quat(q_norm)
        rotation_matrix = r.as_matrix()
        pose_np = np.eye(4)
        pose_np[:3, :3] = rotation_matrix
        pose_np[:3, 3] = [x, y, z]
        
        pcl_location = local_pointcloud[0:3,:]
        pcl_color = torch.from_numpy(local_pointcloud[3:6,:].T).to(self.device).float()
        
        N = local_pointcloud.shape[1]
        one = np.ones((1, N))

        pcl_h = np.vstack((pcl_location, one))
        pcl_h = torch.from_numpy(pcl_h).to(self.device).float()
        b_to_w = torch.from_numpy(pose_np).to(self.device).float()
        c_to_b = torch.tensor([[0, 0, 1, 0],
                                  [-1, 0, 0, 0],
                                  [0, -1, 0, 0],
                                  [0, 0, 0, 1]], dtype=torch.float, device=self.device)

        c_to_w = torch.matmul(b_to_w,c_to_b)
        pcl_h_world = torch.matmul(c_to_w, pcl_h)
        pcl_world = pcl_h_world[:3].T
        pcl_world = torch.hstack((pcl_world,pcl_color))
        
        return pcl_world
    
    def process_data(self, data_file):
        full_path = os.path.join(self.data_folder, data_file)
        data = np.load(full_path)
        pointcloud = data['pointcloud']
        gt_odom = data['gt_odom']
        est_odom = data['est_odom']
        
        gt_rotation = R.from_quat(gt_odom[3:])
        gt_yaw = gt_rotation.as_euler('zyx')[0]
        est_rotation = R.from_quat(est_odom[3:])
        est_yaw = est_rotation.as_euler('zyx')[0]
        gt_odom_result = np.array([gt_odom[0], gt_odom[1], gt_odom[2], gt_yaw])
        est_odom_result = np.array([est_odom[0], est_odom[1], est_odom[2], est_yaw])
        
        pointcloud_local = self.sample_pointcloud(pointcloud)
        pointcloud_global = self.pointcloud_transform_local2world(pointcloud_local, gt_odom).cpu().numpy()
        
        voxel_map_result = self.set_voxel_value(voxel_map, pointcloud_global, gt_odom, est_odom, min_num_points=1)
        
        data.close()
        
        return gt_odom_result, est_odom_result, voxel_map_result

    def generate_map(self):
        
        files = os.listdir(self.data_folder)
        data_files = sorted([file for file in files if file.endswith('.npz')], reverse=True) # time increasing order
        data_file_sampling_rate_for_mapping = 1.0# 10% sampling rate
        num_files_to_sample = int(len(data_files) * data_file_sampling_rate_for_mapping)
        sampled_data_files = []
        stride = len(data_files) // num_files_to_sample
        # Start from the first file and select files at evenly spaced intervals
        for i in range(0, len(data_files), stride):
            sampled_data_files.append(data_files[i])

        # If the number of sampled files is less than num_files_to_sample, append the last file
        if len(sampled_data_files) < num_files_to_sample:
            sampled_data_files.append(data_files[-1])

        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        for data_file  in tqdm(sampled_data_files, desc="Processing data files"):
            full_path = os.path.join(self.data_folder,data_file)
            data = np.load(full_path)
            pointcloud=data['pointcloud'].T # 6XN
            gt_odom = data['gt_odom']
            est_odom = data['est_odom']
            
            pointcloud = self.pointcloud_transform_local2world(pointcloud,gt_odom)
            # Update min and max values for x, y, and z
            min_x = min(min_x, torch.min(pointcloud[:, 0]))
            min_y = min(min_y, torch.min(pointcloud[:, 1]))
            min_z = min(min_z, torch.min(pointcloud[:, 2]))
            max_x = max(max_x, torch.max(pointcloud[:, 0]))
            max_y = max(max_y, torch.max(pointcloud[:, 1]))
            max_z = max(max_z, torch.max(pointcloud[:, 2]))
            
            data.close()
            
        # Print min and max values for x, y, and z
        min_x = min_x.cpu().numpy()
        max_x = max_x.cpu().numpy()
        min_y = min_y.cpu().numpy()
        max_y = max_y.cpu().numpy()
        min_z = min_z.cpu().numpy()
        max_z = max_z.cpu().numpy()
        print("Min x:", min_x)
        print("Max x:", max_x)
        print("Min y:", min_y)
        print("Max y:", max_y)
        print("Min z:", min_z)
        print("Max z:", max_z)

        self.min_xyz = np.array([min_x, min_y, min_z])
        self.max_xyz = np.array([max_x, max_y, max_z])
        self.map_size = np.ceil((self.max_xyz - self.min_xyz) / self.voxel_size).astype(int)
        voxel_map = np.empty(self.map_size, dtype=dict)
        for index in np.ndindex(tuple(self.map_size)):
            voxel_map[index] = {}
        
        gt_odom_list = np.zeros([len(sampled_data_files),4]) # time * x,y,z,yaw
        est_odom_list = np.zeros([len(sampled_data_files),4]) # time * x,y,z,yaw

        # fig_debug_pointcloud_raw = plt.figure()
        # ax_debug_pointcloud_raw = fig_debug_pointcloud_raw.add_subplot(111, projection='3d')
        # norm = plt.Normalize(min_z, max_z)
        # cmap = plt.cm.viridis  # You can choose any colormap you prefer
        for i, data_file in enumerate(tqdm(sampled_data_files, desc="Write voxel map")):
            full_path = os.path.join(self.data_folder,data_file)
            data = np.load(full_path)
            pointcloud=data['pointcloud']
            gt_odom = data['gt_odom']
            est_odom = data['est_odom']
            
            gt_rotation = R.from_quat(gt_odom[3:])
            gt_yaw = gt_rotation.as_euler('zyx')[0]
            est_rotation = R.from_quat(est_odom[3:])
            est_yaw = est_rotation.as_euler('zyx')[0]
            gt_odom_list[i,:]=np.array([gt_odom[0],gt_odom[1],gt_odom[2],gt_yaw])
            est_odom_list[i,:]=np.array([est_odom[0],est_odom[1],est_odom[2],est_yaw])
            pointcloud_local = self.sample_pointcloud(pointcloud)
            pointcloud_global = self.pointcloud_transform_local2world(pointcloud_local,gt_odom).cpu().numpy()
            # scatter = ax_debug_pointcloud_raw.scatter(pointcloud_global[:, 0], pointcloud_global[:, 1], pointcloud_global[:, 2], c=pointcloud_global[:, 2], cmap=cmap, norm=norm, marker='s', alpha=0.5)
            # ax_debug_pointcloud_raw.set_xlabel('X')
            # ax_debug_pointcloud_raw.set_ylabel('Y')
            # ax_debug_pointcloud_raw.set_zlabel('Z')
            voxel_map = self.set_voxel_value(voxel_map,pointcloud_global,gt_odom,est_odom,min_num_points=5) # save voxels which observed more than 5 points ...
            data.close()
        # cbar = plt.colorbar(scatter)
        # cbar.set_label('Z')
        # plt.show()
            
        return voxel_map, gt_odom_list, est_odom_list

    def idx2pose(self, idx_3d):
        return (idx_3d + 0.5) * self.voxel_size + self.min_xyz

    def pose2idx(self, pose):
        local_pose = pose - self.min_xyz
        idx_3d = np.round(local_pose / self.voxel_size - 0.5).astype(int)
        return idx_3d

    def set_voxel_value(self, voxel_map,pointcloud,gt_odom,est_odom,min_num_points=5):
        updated_voxels = set()  # Set to store updated voxel indices
        key = tuple(gt_odom.tolist())  # Convert gt_odom to a tuple to use as a key
        for point_data in pointcloud:
            pose = point_data[0:3]  # Extract pose from point_data
            idx = self.pose2idx(pose)
            if np.all(idx < 0) and np.all(idx >= self.map_size):
                raise ValueError("pointcloud is outside map boundaries")
            
            if key not in voxel_map[idx[0], idx[1], idx[2]]:
                updated_voxels.add((idx[0], idx[1], idx[2]))  # Add updated voxel index to the set
                color = point_data[3:6]
                local_pose = pose - self.idx2pose(idx)  # Transform to local frame
                data = {'local_pose':local_pose, 'color': color, 'gt_odom': gt_odom, 'est_odom': est_odom}
                voxel_map[idx[0], idx[1], idx[2]][key] = data
            elif len(voxel_map[idx[0],idx[1],idx[2]][key]['local_pose']) < 30:
                updated_voxels.add((idx[0], idx[1], idx[2]))  # Add updated voxel index to the set
                color = point_data[3:6]
                local_pose = pose - self.idx2pose(idx)  # Transform to local frame
                voxel_map[idx[0],idx[1],idx[2]][key]['local_pose'] = np.vstack([voxel_map[idx[0],idx[1],idx[2]][key]['local_pose'],local_pose])
                voxel_map[idx[0],idx[1],idx[2]][key]['color'] = np.vstack([voxel_map[idx[0],idx[1],idx[2]][key]['color'],color])
            else:
                updated_voxels.remove((idx[0], idx[1], idx[2])) 
                continue
        
            
        for idx in updated_voxels:    
            keys_to_delete = []
            for key in voxel_map[idx[0], idx[1], idx[2]].keys():
                if voxel_map[idx[0], idx[1], idx[2]][key]['local_pose'].shape[0] < min_num_points:
                    keys_to_delete.append(key)
            # Delete keys after iterating
            for key in keys_to_delete:
                del voxel_map[idx[0], idx[1], idx[2]][key]

        return voxel_map

    def copy_dict_map_voxel_value(self,original_voxel_data,dst_idx):
        self.voxel_map[dst_idx[0],dst_idx[1],dst_idx[2]]=original_voxel_data

    def get_voxel_value_pose(self, pose):
        idx_3d = self.pose2idx(pose)
        if np.all(idx_3d >= 0) and np.all(idx_3d < self.map_size):
            return self.voxel_map[idx_3d[0],idx_3d[1],idx_3d[2]]
        else:
            return None

    def get_voxel_value_idx(self, idx_3d):
        if np.all(idx_3d >= 0) and np.all(idx_3d < self.map_size):
            return self.voxel_map[idx_3d[0],idx_3d[1],idx_3d[2]]
        else:
            raise ValueError("Pose is outside map boundaries")
            
    def get_voxel_value_idx_batch(self, idx_3d_batch):
        if torch.all(idx_3d_batch >= 0) and torch.all(idx_3d_batch <  torch.tensor(self.map_size)):
            voxel_values = self.voxel_map[idx_3d_batch[..., 0], idx_3d_batch[..., 1], idx_3d_batch[..., 2]]

            return voxel_values
        else:
            raise ValueError("Pose is outside map boundaries")
        
    def get_map_size(self):
        return self.map_size
    
    def dict_map_to_numpy_map_with_maskviewpoints(self,in_mask_map,n_viewpoint,num_points):
        gt_map = np.full(np.hstack((in_mask_map.shape,np.array([n_viewpoint,7]))),0.0)
        est_map = np.full(np.hstack((in_mask_map.shape,np.array([n_viewpoint,7]))),0.0)
        points_map_with_view = np.full(np.hstack((in_mask_map.shape,np.array([n_viewpoint,num_points,6]))),0.0)
        out_mask_map = np.full(self.map_size,0.0)
        # Get indices where the mask map value is 1.0
        mask_indices = np.where(in_mask_map == 1.0)

        for idx1, idx2, idx3 in zip(*mask_indices):
            voxel = self.voxel_map[idx1, idx2, idx3]
            valid_key_list, validity = self.check_voxel_validity_from_dict_map(voxel, n_viewpoint=n_viewpoint, num_points=num_points)
            if validity:
                sampled_keys = random.sample(valid_key_list, n_viewpoint)
                out_mask_map[idx1,idx2,idx3]=1.0
                for key_iter, key in enumerate(sampled_keys):
                    value = voxel[key]
                    sampled_indices = np.random.choice(value['local_pose'].shape[0], num_points, replace=False)
                    sampled_local_pose = value['local_pose'][sampled_indices]
                    sampled_color = value['color'][sampled_indices]
                    points_map_with_view[idx1, idx2, idx3, key_iter] = np.hstack((sampled_local_pose, sampled_color))
                    gt_map[idx1, idx2, idx3, key_iter, :] = value['gt_odom']
                    est_map[idx1, idx2, idx3, key_iter, :] = value['est_odom']

        return out_mask_map,gt_map, est_map, points_map_with_view
    
    def dict_map_to_numpy_map_with_maskpoints(self,in_mask_map,n_viewpoint,num_points):
        gt_map = np.full(np.hstack((in_mask_map.shape,np.array([n_viewpoint,7]))),0.0)
        est_map = np.full(np.hstack((in_mask_map.shape,np.array([n_viewpoint,7]))),0.0)
        points_map_with_view = np.full(np.hstack((in_mask_map.shape,np.array([n_viewpoint,num_points,6]))),0.0)
        out_mask_map = np.full(self.map_size,0.0)
        # Get indices where the mask map value is 1.0
        mask_indices = np.where(in_mask_map == 1.0)

        for idx1, idx2, idx3 in zip(*mask_indices):
            voxel = self.voxel_map[idx1, idx2, idx3]
            valid_key_list, validity = self.check_voxel_validity_from_dict_map(voxel, n_viewpoint=n_viewpoint, num_points=num_points)
            if validity:
                sampled_keys = random.sample(valid_key_list, n_viewpoint)
                out_mask_map[idx1,idx2,idx3]=1.0
                for key_iter, key in enumerate(sampled_keys):
                    value = voxel[key]
                    sampled_indices = np.random.choice(value['local_pose'].shape[0], num_points, replace=False)
                    sampled_local_pose = value['local_pose'][sampled_indices]
                    sampled_color = value['color'][sampled_indices]
                    points_map_with_view[idx1, idx2, idx3, key_iter] = np.hstack((sampled_local_pose, sampled_color))
                    gt_map[idx1, idx2, idx3, key_iter, :] = value['gt_odom']
                    est_map[idx1, idx2, idx3, key_iter, :] = value['est_odom']

        return out_mask_map,gt_map, est_map, points_map_with_view
    
    def make_maskmap_from_dict_map_with_constraint(self,n_viewpoint=5,num_points=5):
        out_mask_map = np.full(self.map_size,0.0)
        for idx1 in range(self.map_size[0]):
            for idx2 in range(self.map_size[1]):
                for idx3 in range(self.map_size[2]):
                    voxel = self.voxel_map[idx1,idx2,idx3]
                    _,validity=self.check_voxel_validity_from_dict_map(voxel,n_viewpoint=n_viewpoint,num_points=num_points)
                    if validity:
                        out_mask_map[idx1,idx2,idx3]=1.0
        return out_mask_map

    def check_voxel_validity_from_dict_map(self,in_voxel,n_viewpoint=5,num_points=5):
        valid_key_list = []
        if in_voxel is not {}:
            if len(in_voxel.keys()) >= n_viewpoint:
                # sampled_keys = random.sample( list(in_voxel.keys()), n_viewpoint)
                n_valid_viewpoint=0
                for key_iter, key in enumerate(in_voxel.keys()):
                    # Access the value corresponding to each key
                    value = in_voxel[key]
                    if value['local_pose'].size >= 3*num_points:
                        n_valid_viewpoint=n_valid_viewpoint+1
                        valid_key_list.append(key)
                if n_valid_viewpoint >= n_viewpoint:
                    return valid_key_list,True
                else:
                    return valid_key_list,False
            else:
                return valid_key_list,False
        else:
            return valid_key_list,False
    
    def visualize_mask_map(self,mask_map,map_center_idx=np.array([0,0,0]),odom_list=None):
        # voxel_map_mask = self.voxel_map.make_maskmap(self.voxel_map,n_viewpoint=2,num_points=self.min_num_points)
        # voxel_indices = np.nonzero(voxel_map_mask)  # Get the indices of occupied voxels
        # Plot voxel map with point cloud
        fig_trajectory = plt.figure()
        ax_trajectory = fig_trajectory.add_subplot(111, projection='3d')
        if odom_list != None:
            ax_trajectory.plot(odom_list[:, 0], odom_list[:, 1], odom_list[:, 2], color='r')
        # ax_trajectory.plot(self.gt_odom_list[:, 0], self.gt_odom_list[:, 1], self.gt_odom_list[:, 2], color='g')
        # map_center_idx=mask_map.pose2idx(np.array([0,0,0]))
        x,y,z = np.indices((mask_map.shape[0]+1,mask_map.shape[1]+1,mask_map.shape[2]+1))
        x=(x-map_center_idx[0])*self.voxel_size
        y=(y-map_center_idx[1])*self.voxel_size
        z=(z-map_center_idx[2])*self.voxel_size
        ax_trajectory.voxels(x, y, z, mask_map, edgecolor="k")
        ax_trajectory.set_xlim(x.min(), x.max())
        ax_trajectory.set_ylim(y.min(), y.max())
        ax_trajectory.set_zlim(z.min(), z.max())
        plt.show()

