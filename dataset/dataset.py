import torch
from torch.utils.data.dataset import Dataset
import numpy as np

DEMO_LOC = 'demographics.npy'
DIAGNOSES_LOC = 'diagnoses.npy'
PROCEDURES_MV_LOC = 'procedure_mv.npy'


class MimicData(Dataset):

	def __init__(self, bin_diag=True):
		self.demographics_ = np.load(DEMO_LOC)
		self.diagnoses_ = np.load(DIAGNOSES_LOC)
		self.procedures_mv_ = np.load(PROCEDURES_MV_LOC)
		if bin_diag:
			self.diagnoses_ = self.diagnoses_.astype(bool)

	def __len__(self):
		return len(self.demographics_)

	def __getitem__(self, item):
		return self.demographics_[item],\
			   self.diagnoses_[item],\
			   self.procedures_mv_[item]
