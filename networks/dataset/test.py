from torchvision.utils import make_grid, save_image
import numpy as np
import torch

from sklearn.preprocessing import normalize, OneHotEncoder
data = np.load('labevents_num-1.npy')
print(data.shape)
data = data.transpose((1,0,2))
print(data.shape)
data = [normalize(x, axis=0) for x in data]
data = np.transpose(data, (1,0,2))
print(data.shape)

data = torch.from_numpy(data).unsqueeze_(1)
print(data.shape)
print(data.dtype)

save_image(data[:5000], 'test.png', pad_value=0.5)