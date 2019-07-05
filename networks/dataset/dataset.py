import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os

DEMO_LOC = 'demographics.npy'
DIAGNOSES_LOC = 'diagnoses.npy'
PROCEDURES_MV_LOC = 'procedures_mv.npy'

LOC = os.path.dirname(__file__)

class MimicData(Dataset):

	def __init__(self, bin_diag=True):
		self.demographics_ = torch.from_numpy(np.load(os.path.join(LOC, DEMO_LOC)))
		self.diagnoses_ = torch.from_numpy(np.load(os.path.join(LOC, DIAGNOSES_LOC)))
		self.procedures_mv_ = torch.from_numpy(np.load(os.path.join(LOC, PROCEDURES_MV_LOC))).type(torch.float)
		if bin_diag:
			self.diagnoses_ = self.diagnoses_

	def __len__(self):
		return len(self.demographics_)

	def __getitem__(self, item):
		return self.demographics_[item],\
			   self.diagnoses_[item],\
			   self.procedures_mv_[item]

class MicroShort(Dataset):

	def __init__(self, bin_diag=True):
		self.data_ = torch.from_numpy(np.load(os.path.join(LOC, 'microshort.npy'))).type(torch.float)

	def __len__(self):
		return len(self.data_)

	def __getitem__(self, item):
		return self.data_[item]
