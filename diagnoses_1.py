import settings
import csv
from collections import Counter
import numpy as np
import helper

hamid_start, hamid_pos, hamid_list = helper.hamids()

obs = []

with open(settings.diagnosis_loc) as fin:
	reader = csv.reader(fin, delimiter=',', quotechar='"')
	next(reader, None)
	for line in reader:
		hamid = line[settings.dia_hamid_col]
		if hamid in hamid_pos:
			code = line[settings.dia_icd_col]
			seq = line[settings.dia_seq_col]
			if code != '':
				obs.append((hamid, code, seq))

for i in [3]:
	codes = [x[1] for x in obs]
	codes_counts = Counter(codes)
	codes_list = sorted(set(codes))
	hamid_counts = helper.hamid_count(obs, codes_list)
	obs = [(h,c,s) if hamid_counts[c] >= settings.DIA_MIN_HAMID else (h,c[:i],s) for h, c, s in obs]

codes = [x[1] for x in obs]
codes_counts = Counter(codes)
codes_list = sorted(set(codes))
hamid_counts = helper.hamid_count(obs, codes_list)

codes_pos = {c: i for i, c in enumerate(codes_list)}

seqs = [x[2] for x in obs]
seq_counts = Counter(seqs)
seqs_list = sorted(set(seqs))
seqs_pos = {c: i for i, c in enumerate(seqs_list)}

count_matrix = np.zeros((len(hamid_list), len(codes_list)), dtype=int)

for h, c, s in obs:
	if c in codes_pos:
		i = hamid_pos[h]
		j = codes_pos[c]
		k = int(seqs_pos[s])
		count_matrix[i, j] += 1

with open('out/diagnoses_codes.tsv', 'w+') as fout:
	for i, c in enumerate(codes_list):
		per_hamid_count = np.count_nonzero(count_matrix[:, i])
		fout.write('{}\t{}\t{}\n'.format(c, codes_counts[c], per_hamid_count))

with open('out/diagnoses_counts.tsv', 'w+') as fout:
	for row in count_matrix:
		row = [str(x) for x in row]
		fout.write('\t'.join(row)+'\n')

np.save('out/dims/2-diagnoses_counts', count_matrix)