import os

data_loc = 'data/'

admissions_loc = os.path.join(data_loc, 'ADMISSIONS.csv')
prescriptions_loc = os.path.join(data_loc, 'PRESCRIPTIONS.csv')

pr_type_col = 6
pr_start_col = 4
pr_end_col = 5
pr_name_col = 10#7
pr_hamid_col = 2
pr_strength_col = 13
pr_dose_col = 14
pr_units_col = 15
pr_form_col = 16
pr_form_unit_col = 17