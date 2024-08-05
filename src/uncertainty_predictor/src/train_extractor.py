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

class VoxelMapDataset(Dataset):
    def __init__(self, voxel_map,cfg):
        self.voxel_map = voxel_map
        self.min_num_view = cfg['min_viewpoint']
        self.min_points = cfg['min_points']
        self.num_voxel_per_data = cfg['num_voxel_per_data']
        self.occupied_voxels = self.voxel_map.get_occupied_voxels(min_num_view=self.min_num_view) # observed more than self.min_num_view times at different viewpoints
        

    def __len__(self):
        return len(self.occupied_voxels)

    def __getitem__(self, idx):
        voxel_idx = self.occupied_voxels[idx]
        voxel_data = self.voxel_map.get_voxel_value_idx(voxel_idx)
        view_list = list(voxel_data.keys())
        sampled_view_list = random.sample(view_list,  self.num_voxel_per_data) # sample num_voxel_per_data voxels .
        
        output = {'local_pose': [], 'color': [], 'gt_odom': [], 'est_odom': []}
        for key in sampled_view_list:
            cur_voxel_data = voxel_data[key]
            random_indices = np.random.choice(cur_voxel_data['local_pose'].shape[0], size=self.min_points, replace=False) # sample min_points points in a voxel.
            cur_voxel_data['local_pose'] = cur_voxel_data['local_pose'][random_indices]
            cur_voxel_data['color'] = cur_voxel_data['color'][random_indices]/255.0 # normarlzied
            
            output['local_pose'].append(cur_voxel_data['local_pose'])
            output['color'].append(cur_voxel_data['color'])
            output['gt_odom'].append(cur_voxel_data['gt_odom'])
            output['est_odom'].append(cur_voxel_data['est_odom'])


        return {'local_pose': output['local_pose'], 'color': output['color'], 'gt_odom': output['gt_odom'],'est_odom': output['est_odom']}
    
data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
cfg_folder = data_dir = os.path.join(os.path.dirname(__file__), '..', 'cfg')

map_file_path = os.path.join(data_folder, 'map.pkl')
cfg = utils.load_config(cfg_folder+'/'+'train.yaml')


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
    

dataset = VoxelMapDataset(map.voxel_map,cfg)

class CNN(nn.Module):
    def __init__(self,latent_code_length):
        super(CNN, self).__init__()
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
        
                
# Create a DataLoader for your VoxelMapDataset
batch_size = cfg['batch_size']
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define your loss function and optimizer
model = CNN(latent_code_length=16)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train your CNN
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        # batch     : different voxel idx
        # local_pose: num_view_point * batch * num_points_in_a_voxel * channel of point (x,y,z)
        # color     : num_view_point * batch * num_points_in_a_voxel * channel of point (r,g,b)
        # gt_odom   : num_view_point * batch * channel of data (x,y,z,qx,qy,qz,qw)
        # est_odom  : num_view_point * batch * channel of data (x,y,z,qx,qy,qz,qw)
        local_pose, color, gt_odom,est_odom = data['local_pose'],data['color'],data['gt_odom'],data['est_odom']
        
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
