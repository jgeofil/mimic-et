import settings
import csv
from datetime import datetime
import helper
import numpy as np
from collections import Counter

hamid_start, hamid_pos, hamid_list = helper.hamids()

obs = []

names = {}

with open(settings.micro_loc) as fin:
	reader = csv.reader(fin, delimiter=',', quotechar='"')
	next(reader, None)
	for line in reader:

		hamid = str(line[settings.mcb_hamid_col])
		if hamid in hamid_start:
			code = line[settings.mcb_orgid_col]
			name = line[settings.mcb_name_col]

			start = datetime.strptime(line[settings.mcb_date_col], settings.dtf)
			start = int((start - hamid_start[hamid]).total_seconds()//(3600*settings.INTERVAL_H))

			if start >= 0 and start < settings.MAX_H/settings.INTERVAL_H and code != '':
				obs.append((hamid, code, start))
				names[code] = name


codes = set([x[1] for x in obs])
codes_list = sorted(codes)

codes_counts = Counter(x[1] for x in obs)
hamid_counts = {x: len(set([o[0] for o in obs if o[1] == x])) for x in codes_list}

interval_counts = Counter(x[2] for x in obs)

codes_list = [x for x in codes_list if hamid_counts[x] >= settings.MCB_MIN_HAMID]

codes_pos = {x: i for i, x in enumerate(codes_list)}

with open('out/microbiology_codes.tsv', 'w+') as fout:
	for c in codes_list:
		fout.write('{}\t{}\t{}\t{}\n'.format(c, codes_counts[c], hamid_counts[c], names[c]))

with open('out/microbiology_steps.tsv', 'w+') as fout:
	for c in range(settings.MAX_H//settings.INTERVAL_H):
		fout.write('{}\t{}\n'.format(c, interval_counts[c]))

matrix = np.zeros((len(hamid_list), len(codes_list), len(interval_counts)), dtype=np.bool)

for o in obs:
	hamid = o[0]
	code = o[1]
	start = o[2]
	if code in codes_pos:
		matrix[hamid_pos[hamid], codes_pos[code], start] = 1

np.save('out/dims/3-microbiology_bin', matrix)