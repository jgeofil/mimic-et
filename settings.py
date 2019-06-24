import os

data_loc = 'data/'

dtf = "%Y-%m-%d %H:%M:%S"

MIN_H = 24
MAX_H = 24*21
INTERVAL_H = 24

MCB_MIN_HAMID = 100
PMV_MIN_HAMID = 100

admissions_loc = os.path.join(data_loc, 'ADMISSIONS.csv')
prescriptions_loc = os.path.join(data_loc, 'PRESCRIPTIONS.csv')
procedures_loc = os.path.join(data_loc, 'PROCEDURES_ICD.csv')
procedures_mv_loc = os.path.join(data_loc, 'PROCEDUREEVENTS_MV.csv')
items_loc = os.path.join(data_loc, 'D_ITEMS.csv')
micro_loc = os.path.join(data_loc, 'MICROBIOLOGYEVENTS.csv')

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

po_hamid_col = 2
po_seq_col = 3
po_icd_col = 4

pmv_hamid_col = 2
pmv_start_col = 4
pmv_code_col = 6
pmv_value_col = 7
pmv_unit_col = 8
pmv_loc_col = 9

mcb_hamid_col = 2
mcb_date_col = 3
mcb_orgid_col = 7
mcb_name_col = 8