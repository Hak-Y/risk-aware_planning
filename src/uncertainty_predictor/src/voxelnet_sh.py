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
    def __init__(self, in_channels, out_channels,num_viewpoints,num_points):
        self.out_channels = out_channels
        self.num_view_points=num_viewpoints
        self.num_points=num_points
        super(VFE_layer, self).__init__()
        # Define fully connected layer with batch normalization and ReLU
        self.FCN = nn.Sequential(
            nn.Linear(in_channels, out_channels//2),
            # nn.BatchNorm1d(out_channels//2),
            nn.ReLU(),
            nn.Linear(out_channels//2, out_channels//2),
            # nn.BatchNorm1d(out_channels//2),
            nn.ReLU()
        )

    def forward(self, x):
        original_shape = x.shape
        batch_size= original_shape[0]
        x=x.float()
        x = self.FCN[0](x)
        x = x.view(batch_size, -1, self.out_channels//2)# We'll treat the batch and other dimensions separately for batch normalization
        batch_norm_layer = nn.BatchNorm1d(x.size(1))# BatchNorm1d because we're normalizing along the first dimension
        x = batch_norm_layer(x)# Apply batch normalization 
        x=x.view(*original_shape[:-1],self.out_channels//2)  # Reshape back to the original shape
        # Perform max pooling
        x=self.FCN[1](x)
        x=self.FCN[2](x)
        original_shape = x.shape
        x = x.view(batch_size, -1, self.out_channels//2)# We'll treat the batch and other dimensions separately for batch normalization
        batch_norm_layer = nn.BatchNorm1d(x.size(1))# BatchNorm1d because we're normalizing along the first dimension
        x = batch_norm_layer(x)# Apply batch normalization 
        x=x.view(*original_shape[:-1],self.out_channels//2)  # Reshape back to the original shape
        x=self.FCN[3](x)
        if len(x.shape) ==7: # for contrastsive learning
            x_max_pooling, _ = torch.max(x, dim=5, keepdim=True)
            # Concatenate the results of FC and max pooling
            x_max_pooling_expanded = x_max_pooling.expand(-1, -1, -1, -1, -1,self.num_points, -1)
            x_concatenated = torch.cat((x, x_max_pooling_expanded), dim=6)
        else: # for drift learning
            x_max_pooling, _ = torch.max(x, dim=4, keepdim=True)
            # Concatenate the results of FC and max pooling
            x_max_pooling_expanded = x_max_pooling.expand(-1, -1, -1, -1, self.num_view_points*self.num_points, -1)
            x_concatenated = torch.cat((x, x_max_pooling_expanded), dim=5)
            
        return x_concatenated
    
class VoxelNet(nn.Module):
    def __init__(self,num_viewpoints,num_points,num_channel,feature_length):
        super(VoxelNet, self).__init__()
        self.num_view_points=num_viewpoints
        self.num_points=num_points
        layer1_inchannels=num_channel
        layer1_outchannels=layer1_inchannels*4
        layer2_inchannels=layer1_outchannels
        layer2_outchannels=layer2_inchannels*4
        self.fc1_outchannels=64
        self.fc2_outchannels=32
        self.fc3_outchannels=16
        self.VFE_layer1 = VFE_layer(in_channels=layer1_inchannels, out_channels=layer1_outchannels,num_viewpoints=num_viewpoints,num_points=num_points)
        self.VFE_layer2 = VFE_layer(in_channels=layer2_inchannels, out_channels=layer2_outchannels,num_viewpoints=num_viewpoints,num_points=num_points)
        self.FCN = nn.Sequential(
            nn.Linear(layer2_outchannels, self.fc1_outchannels),  # Adjusted input size
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(self.fc1_outchannels, self.fc2_outchannels),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(self.fc2_outchannels, self.fc3_outchannels)  # Adjusted output size
        )

    def forward(self, x):
        # Pass input through VFE_layer1
        x = self.VFE_layer1(x)
        # Pass output of VFE_layer1 through VFE_layer2
        x = self.VFE_layer2(x)
        # Flatten the tensor before passing through the FCN
        x = self.FCN[0](x)
        original_shape=x.shape
        batch_size= original_shape[0]
        x = x.view(batch_size, -1, x.shape[-1])# We'll treat the batch and other dimensions separately for batch normalization
        batch_norm_layer = nn.BatchNorm1d(x.size(1))# BatchNorm1d because we're normalizing along the first dimension
        x = batch_norm_layer(x)# Apply batch normalization 
        x=x.view(*original_shape[:-1], self.fc1_outchannels)  # Reshape back to the original shape
        x = self.FCN[1](x)
        x = self.FCN[2](x)
        original_shape=x.shape
        x = x.view(batch_size, -1, x.shape[-1])# We'll treat the batch and other dimensions separately for batch normalization
        batch_norm_layer = nn.BatchNorm1d(x.size(1))# BatchNorm1d because we're normalizing along the first dimension
        x = batch_norm_layer(x)# Apply batch normalization 
        x=x.view(*original_shape[:-1], self.fc2_outchannels)  # Reshape back to the original shape
        x = self.FCN[3](x)
        x = self.FCN[4](x)

        if len(x.shape)==7:
            x_max_pooling, _ = torch.max(x, dim=5, keepdim=True)
            x_max_pooling = x_max_pooling.squeeze(5)
        else:
            x_max_pooling, _ = torch.max(x, dim=4, keepdim=True)
            x_max_pooling = x_max_pooling.squeeze(4)
        return x_max_pooling