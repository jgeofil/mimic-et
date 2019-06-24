import settings
import csv
from collections import Counter
import numpy as np
import helper

pro = {}
codes = []
seqs = []

hamid_start, hamid_pos, hamid_list = helper.hamids()

with open(settings.procedures_loc) as fin:
	reader = csv.reader(fin, delimiter=',', quotechar='"')
	next(reader, None)
	for line in reader:
		hamid = line[settings.po_hamid_col]
		if hamid in hamid_pos:
			code = line[settings.po_icd_col][:3]

			seq = line[settings.po_seq_col]

			codes.append(code)
			seqs.append(seq)

			if hamid in pro:
				pro[hamid].append((seq, code))
			else:
				pro[hamid] = [(seq, code)]

code_counts = Counter(codes)
codes = set(codes)
norm_codes = {c: i for i, c in enumerate(codes)}
r_norm_codes = {norm_codes[c]: c for c in norm_codes}
seq_counts = Counter(seqs)
seqs = set(seqs)

count_matrix = np.zeros((len(hamid_list), len(codes)), dtype=int)
order_matrix = np.zeros((len(hamid_list), len(seqs)), dtype=int)
for h in pro:
	i = hamid_pos[h]
	for s, c in pro[h]:
		count_matrix[i, norm_codes[c]] += 1
		order_matrix[i, int(s)-1] = norm_codes[c] + 1

with open('out/procedures_codes.tsv', 'w+') as fout:
	for i in range(len(r_norm_codes)):
		c = r_norm_codes[i]
		per_hamid_count = np.count_nonzero(count_matrix[:, i])
		fout.write('{}\t{}\t{}\n'.format(c, code_counts[c], per_hamid_count))

with open('out/procedures_hamids.tsv', 'w+') as fout:
	for h in hamid_list:
		occ_count = np.count_nonzero(order_matrix[i])
		per_code_count = np.count_nonzero(count_matrix[i])
		fout.write('{}\t{}\t{}\n'.format(h, occ_count, per_code_count))

with open('out/procedures_ordered.tsv', 'w+') as fout:
	for row in order_matrix:
		row = [str(x) for x in row]
		fout.write('\t'.join(row)+'\n')

with open('out/procedures_counts.tsv', 'w+') as fout:
	for row in count_matrix:
		row = [str(x) for x in row]
		fout.write('\t'.join(row)+'\n')
