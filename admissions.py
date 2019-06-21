import settings
from datetime import datetime
from matplotlib import pyplot as plt
from collections import Counter

dtf = "%Y-%m-%d %H:%M:%S"
hours_min = 24
hours_max = 2400

def s_to_hours(val):
	return val/(60*60)

hour_list = []
hamids = []

import csv
with open(settings.admissions_loc) as fin:
	reader = csv.reader(fin, delimiter=',', quotechar='"')
	next(reader, None)
	for line in reader:
		start = datetime.strptime(line[3], dtf)
		end = datetime.strptime(line[4], dtf)
		seconds = abs(end-start).total_seconds()
		hours = s_to_hours(seconds)
		if hours >= hours_min and hours <= hours_max:
			hour_list.append(hours)
			hamids.append(line[2])


plt.hist(hour_list, bins=100)
plt.yscale('log')
plt.show()

print('Total admissions conserved: ', len(hamids))

presc_equi = {}
with open('meta/PRESC_EQUI.tsv') as fin:
	for line in fin:
		line = line.strip().split('\t')
		presc_equi[line[0]] = line[1]

presc_ignore = []
with open('meta/PRESC_IGNORE.tsv') as fin:
	for line in fin:
		line = line.strip()
		presc_ignore.append(line)

presc = []
presc_hamid = []
presc_name_col = 7
hamid_col = 2
units_col = 15

units = []

def clean_name(drug_name):
	return drug_name.replace('"', '')


with open(settings.prescriptions_loc) as fin:
	reader = csv.reader(fin, delimiter=',', quotechar='"')
	next(reader, None)
	for line in reader:
		drug_name = clean_name(line[presc_name_col])
		unit = line[units_col]
		if drug_name not in presc_ignore:
			if drug_name in presc_equi:
				drug_name = presc_equi[drug_name]
			hamid = line[hamid_col]
			presc.append(drug_name)
			presc_hamid.append((drug_name, hamid))
			units.append((drug_name, unit))

presc_hamid = set(presc_hamid)
presc_hamid = [x[0] for x in presc_hamid]

counts = Counter(presc)
counts_hamid = Counter(presc_hamid)

print(counts_hamid)

presc = set(presc)

presc_with_counts = [(x, counts[x], counts_hamid[x]) for x in presc]
presc_with_counts = sorted(presc_with_counts, key=lambda x: x[1])

presc_with_counts = [x for x in presc_with_counts if x[1]>50]
presc_dict = {x[0]: 1 for x in presc_with_counts}

with open('out.tsv', 'w+') as fout:
	for item in presc_with_counts:
		fout.write('{}\t{}\t{}\n'.format(*item))

units = set(units)
units = [x for x in units if x[0] in presc_dict]
units = sorted(units)
with open('units.tsv', 'w+') as fout:
	for line in units:
		fout.write('{}\t{}\n'.format(*line))
