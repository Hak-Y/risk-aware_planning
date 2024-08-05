import numpy as np
import torch
import random
import multiprocessing
from tqdm import tqdm

class VoxelMap:
    def __init__(self, voxel_size, min_xyz, max_xyz):
        self.voxel_size = voxel_size
        self.min_xyz = min_xyz
        self.max_xyz = max_xyz
        self.map_size = np.ceil((max_xyz - min_xyz) / voxel_size).astype(int)
        self.voxel_map = np.empty(self.map_size, dtype=dict)
        for index in np.ndindex(tuple(self.map_size)):
            self.voxel_map[index] = {}

    def idx2pose(self, idx_3d):
        return (idx_3d + 0.5) * self.voxel_size + self.min_xyz

    def pose2idx(self, pose):
        local_pose = pose - self.min_xyz
        idx_3d = np.round(local_pose / self.voxel_size - 0.5).astype(int)
        return idx_3d

    def set_voxel_value(self, pointcloud,gt_odom,est_odom,min_num_points=2):
        updated_voxels = set()  # Set to store updated voxel indices
        key = tuple(gt_odom.tolist())  # Convert gt_odom to a tuple to use as a key
        for point_data in pointcloud:
            pose = point_data[0:3]  # Extract pose from point_data
            idx = self.pose2idx(pose)
            if np.all(idx < 0) and np.all(idx >= self.map_size):
                raise ValueError("pointcloud is outside map boundaries")
            
            if key not in self.voxel_map[idx[0], idx[1], idx[2]]:
                updated_voxels.add((idx[0], idx[1], idx[2]))  # Add updated voxel index to the set
                color = point_data[3:6]
                local_pose = pose - self.idx2pose(idx)  # Transform to local frame
                data = {'local_pose':local_pose, 'color': color, 'gt_odom': gt_odom, 'est_odom': est_odom}
                self.voxel_map[idx[0], idx[1], idx[2]][key] = data
            elif len(self.voxel_map[idx[0],idx[1],idx[2]][key]['local_pose']) < 30:
                # updated_voxels.add((idx[0], idx[1], idx[2]))  # Add updated voxel index to the set
                color = point_data[3:6]
                local_pose = pose - self.idx2pose(idx)  # Transform to local frame
                self.voxel_map[idx[0],idx[1],idx[2]][key]['local_pose'] = np.vstack([self.voxel_map[idx[0],idx[1],idx[2]][key]['local_pose'],local_pose])
                self.voxel_map[idx[0],idx[1],idx[2]][key]['color'] = np.vstack([self.voxel_map[idx[0],idx[1],idx[2]][key]['color'],color])
            else:
                updated_voxels.remove((idx[0], idx[1], idx[2])) # since may points are in the voxel, for given key
                continue
            
        # for idx in updated_voxels:
        #     keys_to_delete = []
        #     for key in self.voxel_map[idx[0], idx[1], idx[2]].keys():
        #         if self.voxel_map[idx[0], idx[1], idx[2]][key]['local_pose'].shape[0] < min_num_points:
        #             keys_to_delete.append(key)
            
        #     # Delete keys after iterating
        #     for key in keys_to_delete:
        #         del self.voxel_map[idx[0], idx[1], idx[2]][key]

            
                
    
    def copy_voxel_value(self,data_in, idx):
        self.voxel_map[idx]=data_in

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
    
    def get_occupied_voxels(self, min_num_points=2,min_num_view=2):
        occupied_voxels = []
        for idx1 in range(self.map_size[0]):
                for idx2 in range(self.map_size[1]):
                    for idx3 in range(self.map_size[2]):
                        voxel = self.voxel_map[idx1,idx2,idx3]
                        if voxel is not {}:
                            if len(voxel) >=min_num_view:
                                occupied_voxels.append(np.array([idx1, idx2, idx3]))
        return occupied_voxels

    def print_occupied_voxels(self, min_num_points=2):
            for idx1 in range(self.map_size[0]):
                for idx2 in range(self.map_size[1]):
                    for idx3 in range(self.map_size[2]):
                        voxel = self.voxel_map[idx1,idx2,idx3]
                        if voxel is not {}:
                            if voxel['point'].shape[0] >= min_num_points:
                                print("Voxel_idx: ",idx1,idx2,idx3, " value: ",voxel)
          
        
    def make_training_map_process(self, idx1, idx2, idx3, voxel_data):
        n_viewpoint = 5
        num_points = 30
        gt_map = np.zeros((n_viewpoint, 7))
        est_map = np.zeros((n_viewpoint, 7))
        points_map_with_view = np.zeros((n_viewpoint, num_points, 6))
        
        if voxel_data and len(voxel_data.keys()) > n_viewpoint:
            sampled_keys = random.sample(list(voxel_data.keys()), n_viewpoint)
            for key_iter, key in enumerate(sampled_keys):
                value = voxel_data[key]
                if len(value['local_pose']) > num_points:
                    sampled_indices = np.random.choice(value['local_pose'].shape[0], num_points, replace=False)
                    sampled_local_pose = value['local_pose'][sampled_indices]
                    sampled_color = value['color'][sampled_indices]
                    points_map_with_view[key_iter] = np.hstack((sampled_local_pose, sampled_color))
                    gt_map[key_iter] = value['gt_odom']
                    est_map[key_iter] = value['est_odom']

        return gt_map, est_map, points_map_with_view

    def make_training_map_parallel(self,map):
        print("Starting map transformation for training...")
        results = []
        total_size = self.map_size[0] * self.map_size[1] * self.map_size[2]
        with multiprocessing.Pool() as pool, tqdm(total=total_size) as pbar:
            for idx in range(total_size):
                idx1 = idx // (self.map_size[1] * self.map_size[2])
                idx2 = (idx // self.map_size[2]) % self.map_size[1]
                idx3 = idx % self.map_size[2]
                results.append(pool.apply_async(self.make_training_map_process, args=(idx1, idx2, idx3, map[idx1, idx2, idx3])))
                pbar.update(1)

        processed_results = [res.get() for res in results]
        print("Map transformation for training completed.")
        return processed_results
    
    def make_traning_map_loop(self,map):
        n_viewpoint=5
        num_points=30
        gt_map = np.full(np.hstack((map.map_size,np.array([n_viewpoint,7]))),0.0)
        est_map = np.full(np.hstack((map.map_size,np.array([n_viewpoint,7]))),0.0)
        points_map_with_view = np.full(np.hstack((map.map_size,np.array([n_viewpoint,num_points,6]))),0.0)
        # points_map_wiou_view = np.full(np.hstack((map.map_size,np.array([n_viewpoint*num_points,6]))),0.0)
        for idx1 in range(map.map_size[0]):
            for idx2 in range(map.map_size[1]):
                for idx3 in range(map.map_size[2]):
                    voxel = map.voxel_map[idx1,idx2,idx3]
                    if voxel is not {}:
                        if len(voxel.keys()) > n_viewpoint:
                            sampled_keys = random.sample( list(voxel.keys()), n_viewpoint)
                            for key_iter, key in enumerate(sampled_keys):
                                # Access the value corresponding to each key
                                value = voxel[key]
                                if len(value['local_pose']) > num_points:
                                    sampled_indices = np.random.choice(value['local_pose'].shape[0], num_points, replace=False)
                                    # Sample rows from the original array using the sampled indices
                                    sampled_local_pose = value['local_pose'][sampled_indices]
                                    sampled_color = value['color'][sampled_indices]
                                    points_map_with_view[idx1,idx2,idx3,key_iter]=np.hstack((sampled_local_pose,sampled_color))
                                    gt_map[idx1,idx2,idx3,key_iter,:]=value['gt_odom']
                                    est_map[idx1,idx2,idx3,key_iter,:]=value['est_odom']
                                    
        # self.points_map_with_view = self.points_map_with_view.reshape(np.hstack((self.map_size,np.array([n_viewpoint*num_points,6]))))
        # print("transforming map for traing DONE...")
        return gt_map,est_map,points_map_with_view
    
    def make_maskmap(self,map,n_viewpoint=5,num_points=5):
        mask_map = np.full(map.map_size,0.0)
        for idx1 in range(map.map_size[0]):
            for idx2 in range(map.map_size[1]):
                for idx3 in range(map.map_size[2]):
                    voxel = map.voxel_map[idx1,idx2,idx3]
                    if voxel is not {}:
                        if len(voxel.keys()) >= n_viewpoint:
                            sampled_keys = random.sample( list(voxel.keys()), n_viewpoint)
                            for key_iter, key in enumerate(sampled_keys):
                                # Access the value corresponding to each key
                                value = voxel[key]
                                if len(value['local_pose']) >= num_points:
                                    mask_map[idx1,idx2,idx3]=1.0
        return mask_map

