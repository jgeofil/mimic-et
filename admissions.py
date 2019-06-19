import settings
from datetime import datetime
from matplotlib import pyplot as plt

dtf = "%Y-%m-%d %H:%M:%S"
hours_min = 24
hours_max = 2400

def s_to_hours(val):
	return val/(60*60)

hour_list = []
hamids = []

with open(settings.admissions_loc) as fin:
	next(fin)
	for line in fin:
		line = line.strip().split(',')

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

presc = []
major = 7
minor = 10
dose = 14

with open(settings.prescriptions_loc) as fin:
	for line in fin:
		line = line.strip().split(',')
		presc.append((line[major], line[minor], line[dose]))

presc = set(presc)
presc = sorted(presc)

with open('out.csv', 'w+') as fout:
	for item in presc:
		fout.write('{}\t{}\t{}\n'.format(*item))