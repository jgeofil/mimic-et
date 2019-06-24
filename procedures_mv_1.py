import settings
import csv
from datetime import datetime
import helper
import numpy as np

from collections import Counter

names = {}
with open(settings.items_loc) as fin:
	reader = csv.reader(fin, delimiter=',', quotechar='"')
	next(reader, None)
	for line in reader:
		link = line[5]
		code = line[1]
		desc = line[2]
		if link == 'procedureevents_mv':
			names[code] = desc

hamid_start, hamid_pos, hamid_list = helper.hamids()

obs = []

with open(settings.procedures_mv_loc) as fin:
	reader = csv.reader(fin, delimiter=',', quotechar='"')
	next(reader, None)
	for line in reader:
		hamid = str(line[settings.pmv_hamid_col])
		if hamid in hamid_start:
			code = line[settings.pmv_code_col]
			start = datetime.strptime(line[settings.pmv_start_col], settings.dtf)
			start = int((start - hamid_start[hamid]).total_seconds()//(3600*settings.INTERVAL_H))

			if start >= 0 and start < settings.MAX_H/settings.INTERVAL_H:
				obs.append((hamid, code, start))

codes = [c[1] for c in obs]
codes_counts = Counter(codes)
codes_list = sorted(set(codes))

hamid_counts = helper.hamid_count(obs, codes_list)

codes_list = [x for x in codes_list if hamid_counts[x] >= settings.PMV_MIN_HAMID]

codes_pos = {c: i for i, c in enumerate(codes_list)}

interval_counts = Counter(x[2] for x in obs)

with open('out/procedure_mv_codes.tsv', 'w+') as fout:
	for c in codes_list:
		fout.write('{}\t{}\t{}\t{}\n'.format(c, codes_counts[c], hamid_counts[c], names[c]))

with open('out/procedure_mv_steps.tsv', 'w+') as fout:
	for c in range(settings.MAX_H//settings.INTERVAL_H):
		fout.write('{}\t{}\n'.format(c, interval_counts[c]))

matrix = np.zeros((len(hamid_list), len(codes_list), len(interval_counts)), dtype=np.bool)

for o in obs:
	hamid = o[0]
	code = o[1]
	start = o[2]
	if code in codes_pos:
		matrix[hamid_pos[hamid], codes_pos[code], start] = 1

np.save('out/dims/procedure_mv_bin', matrix)