import torch
import spconv.pytorch as spconv

def torchTensor2spconvTensor(x):
    # Sum accross channel dimension
    # to find out how many individual points are there
    xNoChans = x.sum(1).to_sparse()
    featuresNc = xNoChans._values()
    indicesNc = xNoChans._indices().transpose(0,1)

    # Create features vector with shape [nPoints, nChannels]
    featuresOut = torch.zeros((len(featuresNc), x.shape[1]))

    # Get all channels from each sparse point, from the original dense tensor x
    for nPoint in range(len(featuresNc)):
        coord = indicesNc[nPoint,...]
        featuresOut[nPoint, :]  = x[coord[0],:,coord[1], coord[2]]

    spatial_shape = x.shape[-2:]
    batch_size = x.shape[0]
    return spconv.SparseConvTensor(featuresOut, indicesNc.int(), spatial_shape, batch_size)

# Create torch tensor with torch convention [batch_number, chann_number, dim1, dim2, ...]
torchTensor = torch.rand(2, 5, 10, 10)
# Create spconvTensor from it
spconvTensor = torchTensor2spconvTensor(torchTensor)
# print(spconvTensor.sparity)
# Measure difference
diff = torch.abs(torchTensor - spconvTensor.dense()).sum()
print(diff.item())