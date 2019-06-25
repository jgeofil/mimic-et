import settings
import csv
from datetime import datetime

obs = []
code_unit = set()

cat_codes = set()
num_codes = set()

with open('out/labevents_filtered.csv') as fin:
	reader = csv.reader(fin, delimiter=',', quotechar='"')
	for line in reader:
		value = line[3]
		vnum = line[4]
		code = line[1]

		try:
			if vnum == '' or float(vnum) != float(value):
				cat_codes.add(code)
			else:
				num_codes.add(code)
		except:
			cat_codes.add(code)


both = cat_codes.intersection(num_codes)
cat_codes = cat_codes.difference(both)
num_codes = num_codes.difference(both)
print(len(cat_codes), len(num_codes), len(both))

with open('out/labevents_filtered.csv') as fin:
	with open('out/labevents_filtered_num.csv', 'w+', newline='') as fnum:
		with open('out/labevents_filtered_cat.csv', 'w+', newline='') as fcat:
			with open('out/labevents_filtered_both.csv', 'w+', newline='') as fboth:
				num_writer = csv.writer(fnum, delimiter=',', quotechar='"')
				cat_writer = csv.writer(fcat, delimiter=',', quotechar='"')
				both_writer = csv.writer(fboth, delimiter=',', quotechar='"')
				reader = csv.reader(fin, delimiter=',', quotechar='"')
				for line in reader:

					value = line[3]
					vnum = line[4]

					code = line[1]

					if code in cat_codes:
						cat_writer.writerow(line)
					elif code in num_codes:
						num_writer.writerow(line)
					else:
						both_writer.writerow(line)
