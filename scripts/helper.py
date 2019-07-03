import settings
from datetime import datetime

def hamids():
	hamid_start = {}
	hamid_pos = []
	with open('out/admissions_start.tsv') as fin:
		for line in fin:
			line = line.strip().split('\t')
			hamid = line[0]
			hamid_start[hamid] = datetime.strptime(line[1], settings.dtf)
			hamid_pos.append(hamid)
	return hamid_start, {x: i for i, x in enumerate(hamid_pos)}, hamid_pos

def hamid_count(obs, codes):
	return {x: len(set([o[0] for o in obs if o[1] == x])) for x in codes}
