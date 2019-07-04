import settings
from datetime import datetime
from matplotlib import pyplot as plt
import csv

def s_to_hours(val):
	return val/(60*60)

gender = {}
dob = {}

with open(settings.patients_loc) as fin:
	reader = csv.reader(fin, delimiter=',', quotechar='"')
	next(reader, None)
	for line in reader:
		s = line[settings.pat_id_col]
		gender[s] = line[settings.pat_gen_col]
		dob[s] = datetime.strptime(line[settings.pat_dob_col], settings.dtf)


hour_list = []
hamids = []
ham_to_sub = {}
age_list = []

with open(settings.admissions_loc) as fin:
	reader = csv.reader(fin, delimiter=',', quotechar='"')
	next(reader, None)
	for line in reader:
		hamid = line[2]
		sub = line[1]

		start = datetime.strptime(line[3], settings.dtf)
		end = datetime.strptime(line[4], settings.dtf)
		seconds = abs(end-start).total_seconds()
		hours = s_to_hours(seconds)

		age = abs(start-dob[sub]).total_seconds()/(3600*24*365)
		if age > 100:
			age = 100
		if age < 18:
			age = 0

		if hours >= settings.MIN_H and hours < settings.MAX_H:
			hour_list.append(hours)
			age_list.append(age)
			hamids.append((hamid, start, gender[sub], age))


plt.hist(age_list, bins=50)
plt.show()

print(len(hamids))

demo_list = []

gender_dict = {'M': 0, 'F': 1}

with open('out/admissions_start.tsv', 'w+') as fout:
	for h, s, g, a in hamids:
		demo_list.append([gender_dict[g], a])
		fout.write('{}\t{}\t{}\t{}\n'.format(h, s, g, a))

from sklearn.preprocessing import normalize, OneHotEncoder

demo_list = normalize(demo_list,)

import numpy as np

np.save('out/dims/demographics.npy', demo_list)
np.save('out/dims/2-admissions', hamids)