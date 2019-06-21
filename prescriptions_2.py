import settings
from collections import Counter
import csv

keep_units = {'mg': 1,
			  'g': 1,
			  'gm': 1,
			  'ml': 1,
			  'MG': 1,
			  'G': 1,
			  'GM': 1,
			  'ML': 1,
			  'mL': 1,
			  'L': 1,
			  'UNIT': 1,
			  'mEq': 1,
			  'mcg': 1,
			  'mmol': 1,
			  'VIAL': 1,
			  'PKT': 1,
			  'SYR': 1,
			  'TAB': 1,
			  'PTCH': 1,
			  'PUFF': 1,
			  'CAP': 1,
			  'TUBE': 1,
			  'NEB': 1,
			  'BTL': 1,
			  'BAG': 1,
			  'DROP': 1,
			  'SPRY': 1,
			  'LOZ': 1,
			  'CADD': 1,
			  'Appl': 1,
			  'AMP': 1,
			  'mcg/hr': 1}

keep_count = 0
invalid_count = 0
reject_count = 0
base_count = 0

with open('out/k_presc_1.tsv') as fin:
	with open('out/r_presc_2.tsv', 'w+', newline='') as freject:
		with open('out/k_presc_2.tsv', 'w+', newline='') as fkeep:
			with open('out/i_presc_2.tsv', 'w+', newline='') as finvalid:
				with open('out/b_presc_2.tsv', 'w+', newline='') as fbase:
					reader = csv.reader(fin, delimiter='\t', quotechar='"')
					reject_writer = csv.writer(freject, delimiter='\t', quotechar='"')
					keep_writer = csv.writer(fkeep, delimiter='\t', quotechar='"')
					invalid_writer = csv.writer(finvalid, delimiter='\t', quotechar='"')
					base_writer = csv.writer(fbase, delimiter='\t', quotechar='"')
					for line in reader:
						units = line[settings.pr_units_col]
						dose = line[settings.pr_dose_col]
						if units in keep_units:
							try:
								if '-' in dose:
									left, right = dose.split('-')
									left, right = float(left), float(right)
									dose_float = (left+right)/2.0
								elif ',' in dose:
									dose = dose.replace(',', '')
									dose_float = float(dose)
								else:
									dose_float = float(dose)

								if units in ['g', 'gm', 'G', 'GM']:
									line[settings.pr_units_col] = 'mg'
									dose_float = dose_float*1000
								if units == 'mcg':
									line[settings.pr_units_col] = 'mg'
									dose_float = dose_float/1000.0
								if units == 'mL':
									line[settings.pr_units_col] = 'ml'
								if units == 'mG':
									line[settings.pr_units_col] = 'mg'
								if units == 'L':
									line[settings.pr_units_col] = 'ml'
									dose_float = dose_float*1000.0

								line[settings.pr_dose_col] = str(dose_float)

								if line[settings.pr_type_col] == 'BASE':
									base_count += 1
									base_writer.writerow(line)
								else:
									keep_count += 1
									keep_writer.writerow(line)
							except:
								invalid_count += 1
								invalid_writer.writerow(line)

						else:
							reject_count += 1
							reject_writer.writerow(line)

print('Kept: ', keep_count)
print('Base: ', base_count)
print('Invalid: ', invalid_count)
print('Rejected: ', reject_count)
