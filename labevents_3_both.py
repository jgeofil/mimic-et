import settings
import csv
from datetime import datetime
import helper
import numpy as np
from collections import Counter

hamid_start, hamid_pos, hamid_list = helper.hamids()

obs = []
code_cats = {}

names = {}
with open(settings.d_lab_items_loc) as fin:
	reader = csv.reader(fin, delimiter=',', quotechar='"')
	next(reader, None)
	for line in reader:
		code = line[1]
		desc = line[2]
		names[code] = desc

with open('out/labevents_filtered_both.csv') as fin:
	with open('out/labevents_both_reject.tsv', 'w+') as fout:
		reader = csv.reader(fin, delimiter=',', quotechar='"')
		writer = csv.writer(fout, delimiter=',', quotechar='"')
		for line in reader:

			hamid = str(line[0])
			code = line[1]

			value = line[3]

			start = datetime.strptime(line[2], settings.dtf)
			start = int((start - hamid_start[hamid]).total_seconds()//(3600*settings.INTERVAL_H))

			if start >= 0 and start < settings.MAX_H/settings.INTERVAL_H:
				try:
					value = float(value)
					obs.append((hamid, code, start, value))
					if code in code_cats:
						code_cats[code].append(value)
					else:
						code_cats[code] = [value]
				except:
					writer.writerow(line)

codes = [x[1] for x in obs]
code_counts = Counter(codes)
codes_list = sorted(set(codes))

hamid_counts = {x: len(set([o[0] for o in obs if o[1] == x])) for x in codes_list}

interval_counts = Counter(x[2] for x in obs)

codes_list = [x for x in codes_list if hamid_counts[x] >= settings.LAB_MIN_HAMID]

codes_pos = {x: i for i, x in enumerate(codes_list)}

with open('out/labevents_both_codes.tsv', 'w+') as fout:
	for c in codes_list:
		nmin = min(code_cats[c])
		nmax = max(code_cats[c])
		avg = np.average(code_cats[c])
		std = np.std(code_cats[c])
		fout.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(c, code_counts[c], hamid_counts[c], names[c], nmin, nmax, avg, std))

with open('out/labevents_both_steps.tsv', 'w+') as fout:
	for c in range(settings.MAX_H//settings.INTERVAL_H):
		fout.write('{}\t{}\n'.format(c, interval_counts[c]))

matrix = np.full((len(hamid_list), len(codes_list), len(interval_counts)), dtype=np.int8, fill_value=np.nan)

for o in obs:
	hamid = o[0]
	code = o[1]
	start = o[2]
	value = o[3]
	if code in codes_pos:
		matrix[hamid_pos[hamid], codes_pos[code], start] = value

np.save('out/dims/3-labevents_num-2', matrix)