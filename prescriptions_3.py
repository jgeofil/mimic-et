import settings
from collections import Counter
import csv

pres_names = []
dosages = {}
values = {}

with open('out/k_presc_2.tsv') as fin:
	reader = csv.reader(fin, delimiter='\t', quotechar='"')
	for line in reader:
		name = line[settings.pr_name_col]
		unit = line[settings.pr_units_col]

		pres_names.append(name)
		if name not in dosages:
			dosages[name] = [unit]
		else:
			dosages[name].append(unit)

dosages = {k: Counter(dosages[k]) for k in dosages}

counts = Counter(pres_names)
names = set(pres_names)
names = sorted(names)

with open('out/k_presc_2.tsv') as fin:
	with open('out/r_presc_3.tsv', 'w+', newline='') as freject:
		with open('out/k_presc_3.tsv', 'w+', newline='') as fkeep:
			reader = csv.reader(fin, delimiter='\t', quotechar='"')
			reject_writer = csv.writer(freject, delimiter='\t', quotechar='"')
			keep_writer = csv.writer(fkeep, delimiter='\t', quotechar='"')
			for line in reader:
				name = line[settings.pr_name_col]
				unit = line[settings.pr_units_col]
				value = line[settings.pr_dose_col]
				ds = dosages[name].most_common(1)[0][0]
				if unit == ds:
					keep_writer.writerow(line)
					if name not in values:
						values[name] = [value]
					else:
						values[name].append(value)
				else:
					reject_writer.writerow(line)

values = {k: Counter(values[k]) for k in values}

with open('out/name_presc_3.tsv', 'w+', newline='') as fnames:
	names_writer = csv.writer(fnames, delimiter='\t', quotechar='"')
	for k in names:
		val_counts = values[k]
		names_writer.writerow((k, counts[k], dosages[k].most_common(1)[0][0], ', '.join([str(val_counts[x]) for x in val_counts])))
