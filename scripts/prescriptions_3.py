import settings
import csv
from datetime import datetime
import helper
import numpy as np
from collections import Counter

hamid_start, hamid_pos, hamid_list = helper.hamids()

obs = []
units = {}

with open('out/k_presc_2.tsv') as fin:
	reader = csv.reader(fin, delimiter='\t', quotechar='"')
	for line in reader:

		hamid = line[settings.pr_hamid_col]
		if hamid in hamid_pos:
			code = line[settings.pr_name_col]
			unit = line[settings.pr_units_col]
			value = line[settings.pr_dose_col]

			date = line[settings.pr_start_col]

			if date != '':

				start = datetime.strptime(line[4], settings.dtf)
				start = int((start - hamid_start[hamid]).total_seconds() // (3600 * settings.INTERVAL_H))

				if start >= 0 and start < settings.MAX_H / settings.INTERVAL_H:
					obs.append((hamid, code, start, value, unit))
					if code in units:
						units[code].append(unit)
					else:
						units[code] = [unit]

unit_common = {c: Counter(units[c]).most_common(1)[0][0] for c in units}

obs = [o for o in obs if o[4] == unit_common[o[1]]]

codes = [x[1] for x in obs]
code_counts = Counter(codes)
codes_list = sorted(set(codes))

hamid_counts = {x: len(set([o[0] for o in obs if o[1] == x])) for x in codes_list}

interval_counts = Counter(x[2] for x in obs)

codes_list = [x for x in codes_list if hamid_counts[x] >= settings.PRE_MIN_HAMID]

codes_pos = {x: i for i, x in enumerate(codes_list)}

values = {c: Counter([o[3] for o in obs if o[1] == c]) for c in codes_list}

obs = [(h, c, s, v) if c in values and values[c][v] >= settings.PR_MIN_OCC else (h, c, s, -1) for h, c, s, v, u in obs]

values = {c: Counter([o[3] for o in obs if o[1] == c]) for c in codes_list}
values_pos = {c: {v: i+1 for i, v in enumerate(values[c])} for c in codes_list}

with open('out/prescriptions_codes.tsv', 'w+') as fout:
	for c in codes_list:
		fout.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(c, code_counts[c], hamid_counts[c], unit_common[c], values[c], values_pos[c]))

with open('out/prescriptions_steps.tsv', 'w+') as fout:
	for c in range(settings.MAX_H//settings.INTERVAL_H):
		fout.write('{}\t{}\n'.format(c, interval_counts[c]))

matrix = np.zeros((len(hamid_list), len(codes_list), len(interval_counts)), dtype=np.int8)

for h, c, s, v in obs:
	if c in codes_pos:
		matrix[hamid_pos[h], codes_pos[c], s] = values_pos[c][v]

np.save('out/dims/3-prescriptions_cat', matrix)
