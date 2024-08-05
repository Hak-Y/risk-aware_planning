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

class VoxelMapDataset(Dataset):
    def __init__(self, data_folder,cfg):
        
        # Check if the map file exists
        if os.path.exists(map_file_path):
            print("Read existing map file...")
            # If the map file exists, load the map object from the file
            with open(map_file_path, 'rb') as f:
                map = pickle.load(f)
            print("Read existing map file DONE...")
        else:
            # If the map file does not exist, calculate the map object
            print("map file does not exist, create the map...")
            map = Map(data_folder, cfg_folder)
            # Save the map object to the file
            with open(map_file_path, 'wb') as f:
                pickle.dump(map, f)
            print("map file creation DONE...")
            
        files = os.listdir(data_folder)
        self.data_folder = data_folder
        self.data_files = sorted([file for file in files if file.endswith('.npz')], reverse=True) # time increasing order
        self.voxel_map = map.voxel_map
        self.min_num_view = cfg['min_viewpoint']
        self.min_points = cfg['min_points']
        self.num_voxel_per_data = cfg['num_voxel_per_data']
        self.occupied_voxels = self.voxel_map.get_occupied_voxels(min_num_view=self.min_num_view) # observed more than self.min_num_view times at different viewpoints
        
        self.p_ray_length_ = 10
        self.p_resolution_x_ = 640
        self.p_resolution_y_ = 480
        self.p_focal_length_ = 386.126953125
        self.p_ray_step_ = self.voxel_map.voxel_size
        self.c_field_of_view_x_ = 2.0 * math.atan(self.p_resolution_x_ / (2.0 * self.p_focal_length_))
        self.c_field_of_view_y_ = 2.0 * math.atan(self.p_resolution_y_ / (2.0 * self.p_focal_length_))
        self.p_downsampling_factor_ = 2
                    
            
        # Downsample to voxel size resolution at max range
        self.c_res_x_ = min(
            math.ceil(self.p_ray_length_ * self.c_field_of_view_x_ / (self.voxel_map.voxel_size * self.p_downsampling_factor_)),
            self.p_resolution_x_)
        self.c_res_y_ = min(
            math.ceil(self.p_ray_length_ * self.c_field_of_view_y_ / (self.voxel_map.voxel_size * self.p_downsampling_factor_)),
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
                        if self.voxel_map.get_voxel_value_pose(current_position) != None and self.voxel_map.get_voxel_value_pose(current_position) != {} :
                            idx = self.voxel_map.pose2idx(current_position)
                            idx = np.array([idx[0],idx[1],idx[2],1])
                            # print("occupied")
                            break
                    result.append(idx)
            result = np.array(result)
            # result = np.unique(result, axis=0)
            return result 
    
    def __getitem__(self, idx):
        cur_data = np.load(os.path.join(self.data_folder,self.data_files[idx]))
        next_data = np.load(os.path.join(self.data_folder,self.data_files[idx+1]))
        command = self.cal_command(cur_x=cur_data['est_odom'], next_x=next_data['est_odom'])
        visible_voxel_idx = self.inverse_raycasting(cur_x = cur_data['gt_odom'])
        delta_x = (next_data['gt_odom']-next_data['est_odom'])-(cur_data['gt_odom']-cur_data['est_odom'])
        return {'visible_voxel_idx': visible_voxel_idx, 'command':command,'delta_x':delta_x}
    

class FeatureExtractor_multiview(nn.Module):
    def __init__(self,latent_code_length):
        super(FeatureExtractor_multiview, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32, 128)  # Adjust the input size dynamically in the forward method
        self.fc2 = nn.Linear(128, latent_code_length)

    def forward(self, x): # batch * channel * num_view_point * num_points
        # do not mix up data for different viewpoint
        batch_size, num_view_point, num_points, channel = x.shape
        latent_feature = []
        for i in range(num_view_point-1):
            cur_x=x[:,:,i,:]
            cur_x = torch.relu(self.conv1(cur_x))
            cur_x = torch.relu(self.conv2(cur_x))
            cur_x = cur_x.view(cur_x.size(0), -1)  # Flatten the tensor
            input_size = cur_x.size(1)  # Calculate the input size dynamically
            self.fc1 = nn.Linear(input_size, 128)  # Adjust the input size dynamically
            cur_x = torch.relu(self.fc1(cur_x))
            cur_x = self.fc2(cur_x)
            latent_feature.append(cur_x)
        latent_feature = torch.stack(latent_feature, dim=1)  # Concatenate the tensors along dimension 1
        return latent_feature

class VFE_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VFE_layer, self).__init__()
        # Define fully connected layer with batch normalization and ReLU
        self.FCN = nn.Sequential(
            nn.Linear(in_channels, out_channels//2),
            nn.BatchNorm1d(out_channels//2),
            nn.ReLU(),
            nn.Linear(out_channels//2, out_channels//2),
            nn.BatchNorm1d(out_channels//2),
            nn.ReLU()
        )

    def forward(self, x):
        x=x.float()
        x_fc = self.FCN(x)
        # Perform max pooling
        x_maxpool, _ = torch.max(x_fc, dim=0, keepdim=True)
        # Concatenate the results of FC and max pooling
        x_maxpool_repeated = x_maxpool.repeat(20, 1)
        x_concat = torch.cat((x_fc, x_maxpool_repeated), dim=1)  # Remove added dimension for concatenation
        return x_concat
    
class VoxelNet(nn.Module):
    def __init__(self):
        super(VoxelNet, self).__init__()
        self.VFE_layer1 = VFE_layer(in_channels=3, out_channels=32)
        self.VFE_layer2 = VFE_layer(in_channels=32, out_channels=128)
        self.FCN = nn.Sequential(
            nn.Linear(128, 64),  # Adjusted input size
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128)  # Adjusted output size
        )

    def forward(self, x):
        # Pass input through VFE_layer1
        x = self.VFE_layer1(x)
        # Pass output of VFE_layer1 through VFE_layer2
        x = self.VFE_layer2(x)
        # Flatten the tensor before passing through the FCN
        x = self.FCN(x)
        x_maxpool, _ = torch.max(x, dim=0, keepdim=True)
        return x_maxpool.squeeze(0)
    
    
class DriftPredictor(nn.Module):
    def __init__(self,latent_code_length):
        super(DriftPredictor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32, 128)  # Adjust the input size dynamically in the forward method
        self.fc2 = nn.Linear(128, latent_code_length)

    def forward(self, x): # batch * channel * num_view_point * num_points
        # do not mix up data for different viewpoint
        batch_size, num_view_point, num_points, channel = x.shape
        latent_feature = []
        for i in range(num_view_point-1):
            cur_x=x[:,:,i,:]
            cur_x = torch.relu(self.conv1(cur_x))
            cur_x = torch.relu(self.conv2(cur_x))
            cur_x = cur_x.view(cur_x.size(0), -1)  # Flatten the tensor
            input_size = cur_x.size(1)  # Calculate the input size dynamically
            self.fc1 = nn.Linear(input_size, 128)  # Adjust the input size dynamically
            cur_x = torch.relu(self.fc1(cur_x))
            cur_x = self.fc2(cur_x)
            latent_feature.append(cur_x)
        latent_feature = torch.stack(latent_feature, dim=1)  # Concatenate the tensors along dimension 1
        return latent_feature
    
def PointinfoNCE(latent_features,hyper_parameter): # batch * num_view_point * latent_code_length
    batch_size, num_view_point, latent_code_length = latent_features.shape 
    loss = 0
    for idx1 in range(batch_size):
        for vp1 in range(num_view_point):
            latent_a = latent_features[idx1][vp1]
            if vp1 + 1 < num_view_point:
                for vp2 in range(vp1+1,num_view_point):
                    latent_b = latent_features[idx1][vp2]
                    # now, latent_a and latent_b are positive pair.
                    positive_numerator=torch.exp(torch.dot(latent_a,latent_b)/hyper_parameter)
            else:
                continue
                    
            negative_denominator = 0
            for idx2 in range(batch_size):
                if idx1==idx2:
                    continue
                for vp3 in range(num_view_point):
                    latent_c = latent_features[idx2][vp3]
                    # now, latent_a and latent_c are negative pair.
                    negative_denominator = negative_denominator + torch.exp(torch.dot(latent_a,latent_c)/hyper_parameter)
                                
            per_positive_pair = -torch.log( positive_numerator/ negative_denominator )
            loss = loss+per_positive_pair
                
    return -loss
                
data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
cfg_folder = data_dir = os.path.join(os.path.dirname(__file__), '..', 'cfg')

map_file_path = os.path.join(data_folder, 'map.pkl')
cfg = utils.load_config(cfg_folder+'/'+'train.yaml')


dataset = VoxelMapDataset(data_folder,cfg)

# Create a DataLoader for your VoxelMapDataset
batch_size = cfg['batch_size']
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define your loss function and optimizer
model_VoxelNet = VoxelNet()
model_DriftPredictor = DriftPredictor(latent_code_length=16)

optimizer = optim.Adam(model_VoxelNet.parameters(), lr=0.001)

# Train your CNN
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        # batch     : num data 
        # visible_voxel_idx : batch X image_size (c_res_x_Xc_res_y_) * 3 (3d voxel idx)
        # command : batch X 4 (x,y,z,yaw)
        # delta_x : batch X 4 (x,y,z,yaw)
        visible_voxel_idx,command,delta_x = data['visible_voxel_idx'],data['command'],data['delta_x']
        visible_voxel_idx_reshaped = visible_voxel_idx.view(batch_size, dataset.c_res_x_, dataset.c_res_y_, -1)
        
        voxel_mask = visible_voxel_idx_reshaped[:,:,:,3]
        visible_voxel_idx_reshaped_wo_mask = visible_voxel_idx_reshaped[:,:,:,:3]
        voxel_data = dataset.voxel_map.get_voxel_value_idx_batch(visible_voxel_idx_reshaped_wo_mask)
        lf_image = torch.zeros((*voxel_mask.shape, 128), dtype=torch.float32)
        for i in range(voxel_data.shape[0]):
            for j in range(voxel_data.shape[1]):
                for k in range(voxel_data.shape[2]):
                    if voxel_mask[i,j,k]==0:
                        continue
                    cur_local_poses = torch.empty((0,6), dtype=torch.float32)
                    for value in voxel_data[i, j, k].values():
                        cur_local_data = torch.cat((torch.tensor(value['local_pose']),torch.tensor(value['color'])),1) # normalizing??
                        cur_local_poses=torch.cat((cur_local_poses,cur_local_data),0)
                        if cur_local_poses.shape[0]>=20:
                            sampled_indices = torch.randperm(cur_local_poses.shape[0])[:20]
                            # Sample 20 elements using the random indices
                            cur_local_poses_sampled = cur_local_poses[sampled_indices]
                            lf_image[i, j, k,:] = model_VoxelNet(cur_local_poses_sampled[:,:3])

        

        local_pose = torch.stack(local_pose, dim=1) # local_pose: batch * num_view_point * num_points_in_a_voxel * channel of point (x,y,z)
        color      = torch.stack(color, dim=1)
        gt_odom    = torch.stack(gt_odom, dim=1)
        est_odom   = torch.stack(est_odom, dim=1)
        
        pt_color=torch.cat((local_pose,color),dim=3).float() #batch * num_view_point * num_points *channel (6)
        pt_color=torch.transpose(pt_color,1,3) # batch * channel * num_view_point * num_points
        
        optimizer.zero_grad()
        outputs = model(pt_color) # batch * num_view_point * latent_code_length
        
        loss = PointinfoNCE(outputs,hyper_parameter=1.0)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')
