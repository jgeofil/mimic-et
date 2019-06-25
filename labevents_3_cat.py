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

with open('out/labevents_filtered_cat.csv') as fin:
	reader = csv.reader(fin, delimiter=',', quotechar='"')
	next(reader, None)
	for line in reader:

		hamid = str(line[0])
		code = line[1]

		value = line[3]

		start = datetime.strptime(line[2], settings.dtf)
		start = int((start - hamid_start[hamid]).total_seconds()//(3600*settings.INTERVAL_H))

		if start >= 0 and start < settings.MAX_H/settings.INTERVAL_H:
			obs.append((hamid, code, start, value))
			if code in code_cats:
				obj = code_cats[code]
				if value not in obj:
					obj[value] = len(obj)
			else:
				code_cats[code] = {value: 0}

codes = [x[1] for x in obs]
code_counts = Counter(codes)
codes_list = sorted(set(codes))

hamid_counts = {x: len(set([o[0] for o in obs if o[1] == x])) for x in codes_list}

interval_counts = Counter(x[2] for x in obs)

codes_list = [x for x in codes_list if hamid_counts[x] >= settings.LAB_MIN_HAMID]

codes_pos = {x: i for i, x in enumerate(codes_list)}

with open('out/labevents_cat_codes.tsv', 'w+') as fout:
	for c in codes_list:
		fout.write('{}\t{}\t{}\t{}\t{}\n'.format(c, code_counts[c], hamid_counts[c], names[c], code_cats[c]))

with open('out/labevents_cat_steps.tsv', 'w+') as fout:
	for c in range(settings.MAX_H//settings.INTERVAL_H):
		fout.write('{}\t{}\n'.format(c, interval_counts[c]))

matrix = np.zeros((len(hamid_list), len(codes_list), len(interval_counts)), dtype=np.int8)

for o in obs:
	hamid = o[0]
	code = o[1]
	start = o[2]
	value = o[3]
	if code in codes_pos:
		matrix[hamid_pos[hamid], codes_pos[code], start] = code_cats[code][value]

np.save('out/dims/3-labevents_cat', matrix)