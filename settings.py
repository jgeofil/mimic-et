import os

data_loc = 'data/'

dtf = "%Y-%m-%d %H:%M:%S"

MIN_H = 24
MAX_H = 24*21
INTERVAL_H = 24

MCB_MIN_HAMID = 500
LAB_MIN_HAMID = 500
PRE_MIN_HAMID = 500

PMV_MIN_HAMID = 500
DIA_MIN_HAMID = 1500
PRO_MIN_HAMID = 500
CPT_MIN_HAMID = 500

PR_MIN_OCC = 50

admissions_loc = os.path.join(data_loc, 'ADMISSIONS.csv')
prescriptions_loc = os.path.join(data_loc, 'PRESCRIPTIONS.csv')
procedures_loc = os.path.join(data_loc, 'PROCEDURES_ICD.csv')
procedures_mv_loc = os.path.join(data_loc, 'PROCEDUREEVENTS_MV.csv')
items_loc = os.path.join(data_loc, 'D_ITEMS.csv')
micro_loc = os.path.join(data_loc, 'MICROBIOLOGYEVENTS.csv')
d_icd_pro_loc = os.path.join(data_loc, 'D_ICD_PROCEDURES.csv')
d_lab_items_loc = os.path.join(data_loc, 'D_LABITEMS.csv')
labevents_loc = os.path.join(data_loc, 'LABEVENTS.csv')
patients_loc = os.path.join(data_loc, 'PATIENTS.csv')
diagnosis_loc = os.path.join(data_loc, 'DIAGNOSES_ICD.csv')
diagnosis_item_loc = os.path.join(data_loc, 'D_ICD_DIAGNOSES.csv')
cpt_loc = os.path.join(data_loc, 'CPTEVENTS.csv')

cpt_hamid_col = 2
cpt_date_col = 4
cpt_code_col = 5
cpt_seq_col = 8
cpt_name_col = 10

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

lab_hamid_col = 2
lab_code_col = 3
lab_date_col = 4
lab_val_col = 5
lab_num_col = 6
lab_unit_col = 7

pat_id_col = 1
pat_gen_col = 2
pat_dob_col = 3

dia_hamid_col = 2
dia_seq_col = 3
dia_icd_col = 4