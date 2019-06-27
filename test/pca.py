from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt

data = np.load('../out/dims/2-diagnoses_counts.npy')

print(np.max(data))