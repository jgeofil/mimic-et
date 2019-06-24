import settings
from datetime import datetime
from matplotlib import pyplot as plt
import csv


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
		if hours >= settings.MIN_H and hours < settings.MAX_H:
			hour_list.append(hours)
			hamids[line[2]] = start

plt.hist(hour_list, bins=10)
plt.show()

print(len(hamids))

with open('out/admissions_start.tsv', 'w+') as fout:
	for h in hamids:
		fout.write('{}\t{}\n'.format(h, hamids[h]))
