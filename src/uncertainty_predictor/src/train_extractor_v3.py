import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from map import Map 
import numpy as np
import pickle
import random
import utils
import math
from scipy.spatial.transform import Rotation as R
from voxelmap import VoxelMap
from pointnet import PointNetfeatkd, PointNetfeatkd_resized
from voxelnext import VoxelResBackBone8xVoxelNeXt
from evidence_network import EvidenceNetwork, evidential_regresssion_loss
from Infonce import InfoNCE
import torch.autograd
from tensorboardX import SummaryWriter

torch.autograd.set_detect_anomaly(True)

class VoxelMapDataset(Dataset):
    def __init__(self, data_folder,batch_size):
        map_file_path = os.path.join(data_folder, 'map.pkl')
        cfg_folder  = os.path.join(os.path.dirname(__file__), '..', 'cfg')
        cfg = utils.load_config(cfg_folder+'/'+'train.yaml')
        
        # Check if the map file exists
        if os.path.exists(map_file_path):
            print("Read existing map file...")
            # If the map file exists, load the map object from the file
            with open(map_file_path, 'rb') as f:
                self.map_data = pickle.load(f)
            print("Read existing map file DONE...")
        else:
            # If the map file does not exist, calculate the map object
            print("map file does not exist, create the map...")
            self.map_data = Map(data_folder, cfg_folder)
            # Save the map object to the file
            with open(map_file_path, 'wb') as f:
                pickle.dump(self.map_data, f)
            print("map file creation DONE...")
            
        files = os.listdir(data_folder)
        self.data_folder = data_folder
        self.data_files = sorted([file for file in files if file.endswith('.npz')], reverse=True) # time increasing order
        self.min_num_view = cfg['min_viewpoint']
        self.min_points = cfg['min_points']
        self.num_voxel_per_data = cfg['num_voxel_per_data']
        # self.occupied_voxels = self.voxel_map.get_occupied_voxels(min_num_view=self.min_num_view) # observed more than self.min_num_view times at different viewpoints
        
        self.p_ray_length_ = 5
        self.p_resolution_x_ = 640
        self.p_resolution_y_ = 480
        self.p_focal_length_ = 386.126953125
        self.p_ray_step_ = self.map_data.voxel_size
        self.c_field_of_view_x_ = 2.0 * math.atan(self.p_resolution_x_ / (2.0 * self.p_focal_length_))
        self.c_field_of_view_y_ = 2.0 * math.atan(self.p_resolution_y_ / (2.0 * self.p_focal_length_))
        self.p_downsampling_factor_ = 2
        
        
        # Downsample to voxel size resolution at max range
        self.c_res_x_ = min(
            math.ceil(self.p_ray_length_ * self.c_field_of_view_x_ / (self.map_data.voxel_size * self.p_downsampling_factor_)),
            self.p_resolution_x_)
        self.c_res_y_ = min(
            math.ceil(self.p_ray_length_ * self.c_field_of_view_y_ / (self.map_data.voxel_size * self.p_downsampling_factor_)),
            self.p_resolution_y_)

    def __len__(self):
        # Count the number of .npz files in the data_folder
        # -1 since __getitem__ use next time step data. so last data can not be used alone.
        num_files = len(self.data_files) -1
        return num_files

    def cal_command(self,cur_x, next_x):
        # Extract translation and rotation components from the provided data
        cur_translation = cur_x[:3]
        cur_rotation = R.from_quat(cur_x[3:])

        next_translation = next_x[:3]
        next_rotation = R.from_quat(next_x[3:])

        # Compute the relative rotation of next_x with respect to cur_x
        relative_rotation = cur_rotation.inv() * next_rotation

        # Rotate the translation vector of next_x by the inverse of the rotation of cur_x
        relative_translation = cur_rotation.inv().apply(next_translation - cur_translation)

        # Compute the yaw angle of cur_x and next_x
        cur_yaw = cur_rotation.as_euler('zyx')[0]
        next_yaw = next_rotation.as_euler('zyx')[0]

        # Compute the yaw angle difference between the two poses
        relative_yaw = next_yaw - cur_yaw
            
        # Ensure yaw is within [-pi, pi] range
        relative_yaw = (relative_yaw + np.pi) % (2 * np.pi) - np.pi
        relative_pose =np.concatenate((relative_translation, [relative_yaw]))
        
        # print("cur_x:", cur_x[:3],cur_yaw)
        # print("next_x:", next_x[:3],next_yaw)
        # print("relative_x:", relative_pose)
        
        # Return relative x, y, z, and yaw
        return relative_pose

    
    def inverse_raycasting(self, cur_x):
        result = []
        orientation = cur_x[3:]
        for i in range(self.c_res_x_):
            for j in range(self.c_res_y_):
                relative_x = i / (self.c_res_x_ - 1.0)
                relative_y = j / (self.c_res_y_ - 1.0)
                camera_direction = np.array([
                    self.p_focal_length_,
                    (0.5 - relative_x) * self.p_resolution_x_,
                    (0.5 - relative_y) * self.p_resolution_y_
                ]) / np.linalg.norm([
                    self.p_focal_length_,
                    (0.5 - relative_x) * self.p_resolution_x_,
                    (0.5 - relative_y) * self.p_resolution_y_
                ])
                orientation_rotation = R.from_quat(orientation)
                orientation_matrix = orientation_rotation.as_matrix()
                direction = np.dot(orientation_matrix, camera_direction)
                distance = 0.0
                idx=np.array([0, 0, 0, 0]) # last digit is mask.
                while distance < self.p_ray_length_:
                    current_position = cur_x[:3] + distance * direction
                    distance += self.p_ray_step_
                    # Check if voxel occupied
                    if self.map_data.get_voxel_value_pose(current_position) != None and self.map_data.get_voxel_value_pose(current_position) != {} :
                        idx = self.map_data.pose2idx(current_position)
                        idx = np.array([idx[0],idx[1],idx[2],1])
                        # print("occupied")
                        break
                result.append(idx)
        result = np.array(result)
        # result = np.unique(result, axis=0)
        return result 
    
    def get_local_voxelmap_mask(self,cur_x):
        # decide local map size for given ray length and fov
        # local_map_len_x = (self.p_ray_length_* math.cos(self.c_field_of_view_x_/2))
        local_map_len_x = self.p_ray_length_
        local_map_len_y = (self.p_ray_length_* math.sin(self.c_field_of_view_x_/2))*2
        local_map_len_z = (self.p_ray_length_* math.sin(self.c_field_of_view_y_/2))*2
        max_xyz = np.array([local_map_len_x, local_map_len_y/2, local_map_len_z/2])
        min_xyz = np.array([0.0, -local_map_len_y/2, -local_map_len_z/2])
        # make local map considering local map size and voxel size
        local_voxel_map = Map(voxel_size=self.map_data.voxel_size,min_xyz=min_xyz,max_xyz=max_xyz)
        local_voxel_map_mask = np.zeros(local_voxel_map.voxel_map.shape)
        
        orientation = cur_x[3:]
        for i in range(self.c_res_x_):
            for j in range(self.c_res_y_):
                relative_x = i / (self.c_res_x_ - 1.0)
                relative_y = j / (self.c_res_y_ - 1.0)
                camera_direction = np.array([
                    self.p_focal_length_,
                    (0.5 - relative_x) * self.p_resolution_x_,
                    (0.5 - relative_y) * self.p_resolution_y_
                ]) / np.linalg.norm([
                    self.p_focal_length_,
                    (0.5 - relative_x) * self.p_resolution_x_,
                    (0.5 - relative_y) * self.p_resolution_y_
                ])
                orientation_rotation = R.from_quat(orientation)
                orientation_matrix = orientation_rotation.as_matrix()
                direction = np.dot(orientation_matrix, camera_direction)
                distance = 0.0
                while distance < self.p_ray_length_:
                    current_position = cur_x[:3] + distance * direction
                    distance += self.p_ray_step_
                    # Check if voxel occupied
                    original_voxel_data = self.map_data.get_voxel_value_pose(current_position)
                    if original_voxel_data != None and original_voxel_data != {} : # occupied
                        local_pose = current_position - cur_x[:3]
                        local_pose_local_frame = np.dot(orientation_matrix.T, local_pose)
                        local_idx = local_voxel_map.pose2idx(local_pose_local_frame)
                        local_voxel_map.copy_dict_map_voxel_value(original_voxel_data,local_idx)
                        local_voxel_map_mask[local_idx[0],local_idx[1],local_idx[2]]=1.0
                        break
        
        return local_voxel_map, local_voxel_map_mask
        
    def __getitem__(self, idx):
        cur_data = np.load(os.path.join(self.data_folder,self.data_files[idx]))
        next_data = np.load(os.path.join(self.data_folder,self.data_files[idx+1]))
        command = self.cal_command(cur_x=cur_data['est_odom'], next_x=next_data['est_odom'])
        # visible_voxel_idx = self.inverse_raycasting(cur_x = cur_data['gt_odom'])
        local_voxel_map, local_voxel_map_raycasting_mask =self.get_local_voxelmap_mask(cur_x = cur_data['gt_odom'])
        # local_voxel_map.visualize_mask_map(mask_map=local_voxel_map_raycasting_mask)
        out_mask_map,gt_map,est_map,points_map_with_view=local_voxel_map.dict_map_to_numpy_map_with_maskviewpoints(in_mask_map=local_voxel_map_raycasting_mask,n_viewpoint=2,num_points=10)
        points_map_with_view[:, :, :, :, :, :3] /= (self.map_data.voxel_size/2) #-1 to 1
        points_map_with_view[:, :, :, :, :, 3:] /= 255.0 #0 to 1
        # local_voxel_map.visualize_mask_map(mask_map=out_mask_map)
        delta_x = (next_data['gt_odom']-next_data['est_odom'])-(cur_data['gt_odom']-cur_data['est_odom'])
        # return {'gt_map': gt_map,'est_map':est_map,'points_map_with_view':points_map_with_view,'local_voxel_map_mask':out_mask_map, 'command':command,'delta_x':delta_x}
        return {'gt_odom':cur_data['gt_odom'],'points_map_with_view':points_map_with_view, 'command':command,'delta_x':delta_x}
         

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_folder = os.path.join(os.path.dirname(__file__), '..', 'valid_data/data_gps/dataset2')
    cfg_folder  = os.path.join(os.path.dirname(__file__), '..', 'cfg')
    dataset = VoxelMapDataset(data_folder,cfg_folder)
    cfg = utils.load_config(cfg_folder+'/'+'train.yaml')
    # Create a DataLoader for your VoxelMapDataset
    batch_size = cfg['batch_size']
    min_num_voxel_per_view = cfg['min_num_voxel_per_view']
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define your loss function and optimizer
    pointcloud_encoder = PointNetfeatkd_resized(k=6, global_feat = True, feature_transform = False).to(device) # point-wise featuer length (x,y,z,r,g,b)
    localmap_encoder = VoxelResBackBone8xVoxelNeXt(input_channels=64).to(device) # 64: voxel-wise feature length
    map_dim=32 # map_dim: encoded local_map feature length
    command_dim = 4 # vel_x,vel_y,vel_z,yawrate
    drift_predictor=EvidenceNetwork(x_dim=map_dim+command_dim, y_dim=3).to(device) # y_dim : predicted x,y,z error
    PointinfoNCE = InfoNCE()

    localmap_encoder_optimizer = optim.Adam(localmap_encoder.parameters(), lr=0.001)
    pointcloud_encoder_optimizer = optim.Adam(pointcloud_encoder.parameters(), lr=0.001)
    drift_predictor_optimizer = torch.optim.Adam(drift_predictor.parameters(), lr=5e-4)

    # Wrap model with DataParallel
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs!")
    #     pointcloud_encoder = torch.nn.DataParallel(pointcloud_encoder)
    #     localmap_encoder = torch.nn.DataParallel(localmap_encoder)
    
    writer = SummaryWriter()
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            # points_map_with_view: batch * local_map_size_x *local_map_size_y *local_map_size_z * num_viewpoionts * num_points * channels
            # command : batch X 4 (x,y,z,yaw)
            # delta_x : batch X 4 (x,y,z,yaw)
            gt_odom, points_map_with_view,command,delta_x = data['gt_odom'], data['points_map_with_view'], data['command'],data['delta_x']
            gt_odom,points_map_with_view, command, delta_x = [t.to(device) for t in data.values()]

            tmp = points_map_with_view.to_sparse(sparse_dim = 4)
            voxel_coords = tmp.indices().transpose(0,1)
            voxel_values = tmp.values()

            all_batch_valid = True
            for batch_idx in range(0,batch_size):
                num_data_batch = (voxel_coords[:, 0] == batch_idx).sum().item()
                if num_data_batch < min_num_voxel_per_view:
                    all_batch_valid = False
                    continue
            if not all_batch_valid:
                print("batch data invalid. continue")
                continue    

            ## LOCALIZATION DRFIT START ##
            # merge viewpoint to batch, and split it later
            points_map_without_view_reshaped = voxel_values.view(-1,voxel_values.shape[-1],voxel_values.shape[-2]*voxel_values.shape[-3]) # -1 * num_points * channel
            latent_feature_local_map,_,_ =  pointcloud_encoder(points_map_without_view_reshaped.float())
            sparse_shape = points_map_with_view.shape[1:4]
            batch_dict = {'voxel_features': latent_feature_local_map,
                          'voxel_coords':voxel_coords,
                          'batch_size':batch_size,
                          'sparse_shape':sparse_shape}
            
            encoded_localmap = localmap_encoder(batch_dict) 
            encoded_localmap=encoded_localmap.dense().squeeze()
            encoded_localmap_with_state = torch.cat([encoded_localmap,command],dim=1)
            output = drift_predictor(encoded_localmap_with_state.float())
            # mean_predicted_drift = output['mu']
            drift_loss = evidential_regresssion_loss(delta_x[:,:3], output, 1e-2)
            ## LOCALIZATION DRFIT START ##

            ## CONSTRATSTIVE START ##
            points_map_with_view_reshaped = voxel_values.view(-1,voxel_values.shape[-1],voxel_values.shape[-2]) # -1 * num_hannel * points per voxels
            latent_feature_local_map_wview,_,_  = pointcloud_encoder(points_map_with_view_reshaped.float())
            latent_feature_local_map_wview=latent_feature_local_map_wview.view(-1,points_map_with_view.shape[4],latent_feature_local_map_wview.shape[-1])
            contrastsive_loss = 0
            for a in range(0,latent_feature_local_map_wview.shape[1]):
                data_view_a = latent_feature_local_map_wview[:,a,:]
                for b in range(a+1,latent_feature_local_map_wview.shape[1]):
                    data_view_b = latent_feature_local_map_wview[:,b,:]
                    loss_per_viewpair = PointinfoNCE(query = data_view_a,positive_key = data_view_b)
                    contrastsive_loss = contrastsive_loss + loss_per_viewpair
            ## CONSTRATSTIVE END ##
            
            localmap_encoder_optimizer.zero_grad()
            drift_predictor_optimizer.zero_grad()
            drift_loss.backward()
            drift_predictor_optimizer.step()
            localmap_encoder_optimizer.step()

            pointcloud_encoder_optimizer.zero_grad()
            contrastsive_loss.backward()
            pointcloud_encoder_optimizer.step()
            
  

            running_loss += drift_loss+contrastsive_loss
            if i % 10 == 0:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, drift_loss: {drift_loss:.3f}, contrastsive_loss: {contrastsive_loss:.3f}')
                running_loss = 0.0

            # Inside the training loop
            writer.add_scalar('Loss/Total', running_loss, epoch * len(dataloader) + i)
            writer.add_scalar('Loss/Drift', drift_loss, epoch * len(dataloader) + i)
            writer.add_scalar('Loss/Contrastive', contrastsive_loss, epoch * len(dataloader) + i)    
           

    print('Finished Training')
    writer.close()

if __name__ == '__main__':
    main()

