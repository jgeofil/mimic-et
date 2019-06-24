import settings
import csv
from datetime import datetime
import helper
import numpy as np

from collections import Counter

with open('out/procedures_mv_codes.tsv', 'w+') as fout:
	with open(settings.items_loc) as fin:
		reader = csv.reader(fin, delimiter=',', quotechar='"')
		next(reader, None)
		for line in reader:
			link = line[5]
			code = line[1]
			desc = line[2]
			if link == 'procedureevents_mv':
				fout.write('{}\t{}\n'.format(code, desc))
hours_max = 600

dtf = "%Y-%m-%d %H:%M:%S"

hamid_start, hamid_pos, hamid_list = helper.hamids()

pro = []

c_hamid = {}
h_codes = {}

num_codes = set()
bin_codes = set()

with open(settings.procedures_mv_loc) as fin:
	reader = csv.reader(fin, delimiter=',', quotechar='"')
	next(reader, None)
	for line in reader:
		hamid = str(line[settings.pmv_hamid_col])
		if hamid in hamid_start:
			code = line[settings.pmv_code_col]
			try:
				value = int(line[settings.pmv_value_col])
			except:
				value = float(line[settings.pmv_value_col])
			unit = line[settings.pmv_unit_col]
			start = datetime.strptime(line[settings.pmv_start_col], dtf)
			start = int((start - hamid_start[hamid]).total_seconds()//3600)

			location = line[settings.pmv_loc_col]

			if unit == 'min':
				pass
			elif unit == 'hour':
				value = value*60
			elif unit == 'day':
				value = value*24*60
			elif unit == 'None':
				value = 1
			else:
				raise ValueError

			if start >= 0 and start < hours_max:
				pro.append((hamid, code, start, value, unit))

				if unit == 'None':
					bin_codes.add(code)
				else:
					num_codes.add(code)

				if code in c_hamid:
					c_hamid[code].append(hamid)
				else:
					c_hamid[code] = [hamid]

				if hamid in h_codes:
					h_codes[hamid].append(code)
				else:
					h_codes[hamid] =[code]

bin_codes_list = [c for c in bin_codes]
bin_codes_pos = {c: i for i, c in enumerate(bin_codes_list)}

num_codes_list = [c for c in num_codes]
num_codes_pos = {c: i for i, c in enumerate(num_codes_list)}

with open('out/procedures_mv_bin.tsv', 'w+') as fbin:
	for c in bin_codes_list:
		fbin.write('{}\t{}\t{}\n'.format(c, len(c_hamid[c]), len(set(c_hamid[c]))))
with open('out/procedures_mv_num.tsv', 'w+') as fnum:
	for c in num_codes_list:
		fnum.write('{}\t{}\t{}\n'.format(c, len(c_hamid[c]), len(set(c_hamid[c]))))
with open('out/procedures_mv_hamids.tsv', 'w+') as fbin:
	for h in hamid_list:
		if h in h_codes:
			fbin.write('{}\t{}\t{}\n'.format(h, len(h_codes[h]), len(set(h_codes[h]))))
		else:
			fbin.write('{}\t{}\t{}\n'.format(h, 0, 0))


bin_matrix = np.zeros((len(hamid_list), len(bin_codes), hours_max), dtype=np.int8)
for p in pro:
	code = p[1]
	hamid = p[0]
	if code in bin_codes:
		bin_matrix[hamid_pos[hamid], bin_codes_pos[code], p[2]] = 1

np.save('out/procedures_mv_bin_cube', bin_matrix)

num_matrix = np.zeros((len(hamid_list), len(num_codes), hours_max), dtype=np.int32)
for p in pro:
	code = p[1]
	hamid = p[0]
	if code in bin_codes:
		bin_matrix[hamid_pos[hamid], bin_codes_pos[code], p[2]] = p[3]

np.save('out/procedures_mv_num_cube', num_matrix)

#with open('out/procedures_mv_num_triples.tsv', 'w+') as fnum: