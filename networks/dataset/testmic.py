from torchvision.utils import make_grid, save_image
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.preprocessing import normalize, OneHotEncoder
data = np.load('3-microbiology_bin.npy').astype(float)
data = np.array([x for x in data if sum(sum(x)) >= 1]).astype(float)

data = torch.from_numpy(data)
data = F.pad(input=data, pad=(6, 5, 11, 11), mode='constant', value=0)
data = data.unsqueeze_(1)
print(data.shape)
print(data.dtype)



save_image(data[:5000], 'test.png', pad_value=0.5)

data = data.numpy()
np.save('microshort.npy', data)