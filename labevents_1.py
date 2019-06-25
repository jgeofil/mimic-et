import settings
import csv
import helper

hamid_start, hamid_pos, hamid_list = helper.hamids()

with open(settings.labevents_loc) as fin:
	with open('out/labevents_filtered.csv', 'w+', newline='') as fkeep:
		writer = csv.writer(fkeep, delimiter=',', quotechar='"')
		reader = csv.reader(fin, delimiter=',', quotechar='"')
		next(reader, None)
		for line in reader:
			hamid = str(line[settings.lab_hamid_col])

			if hamid in hamid_start and line[settings.lab_date_col] != '':
				writer.writerow(line[2:8])


