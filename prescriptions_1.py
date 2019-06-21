import settings
from collections import Counter
import csv

MIN_OCC = 500

name_counts = {}

with open(settings.prescriptions_loc) as fin:
	reader = csv.reader(fin, delimiter=',', quotechar='"')
	next(reader, None)
	for line in reader:
		name = line[settings.pr_name_col]
		hamid = line[settings.pr_hamid_col]
		if name in name_counts:
			name_counts[name].add(hamid)
		else:
			name_counts[name] = set([hamid])

name_counts = {k: len(name_counts[k]) for k in name_counts}

keep = {k: 1 for k in name_counts if name_counts[k] >= MIN_OCC and k != ''}
reject = {k: 1 for k in name_counts if name_counts[k] < MIN_OCC or k == ''}

with open(settings.prescriptions_loc) as fin:
	with open('out/r_presc_1.tsv', 'w+', newline='') as freject:
		with open('out/k_presc_1.tsv', 'w+', newline='') as fkeep:
			reader = csv.reader(fin, delimiter=',', quotechar='"')
			reject_writer = csv.writer(freject, delimiter='\t', quotechar='"')
			keep_writer = csv.writer(fkeep, delimiter='\t', quotechar='"')
			next(reader, None)
			for line in reader:
				name = line[settings.pr_name_col]
				if name in keep:
					keep_writer.writerow(line)
				else:
					reject_writer.writerow(line)

print("Kept: ", len(keep))
print("Rejected: ", len(reject))


