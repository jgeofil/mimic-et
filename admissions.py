import settings
from datetime import datetime
from matplotlib import pyplot as plt
import csv

hours_min = 24
hours_max = 600

def s_to_hours(val):
	return val/(60*60)

hour_list = []
hamids = {}

with open(settings.admissions_loc) as fin:
	reader = csv.reader(fin, delimiter=',', quotechar='"')
	next(reader, None)
	for line in reader:
		start = datetime.strptime(line[3], settings.dtf)
		end = datetime.strptime(line[4], settings.dtf)
		seconds = abs(end-start).total_seconds()
		hours = s_to_hours(seconds)
		if hours >= hours_min and hours < hours_max:
			hour_list.append(hours)
			hamids[line[2]] = start


plt.hist(hour_list, bins=10)
#plt.yscale('log')
plt.show()

print(len(hamids))

with open('out/admissions_start.tsv', 'w+') as fout:
	for h in hamids:
		fout.write('{}\t{}\n'.format(h, hamids[h]))
