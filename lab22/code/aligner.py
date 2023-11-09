"""G2P Aligner"""

__author__ = """Hahn Koo (hahn.koo@sjsu.edu)"""

import sys, pickle

def align(g, p, sub, ins, dlt, gcd, pcd, ins_weight=0, dlt_weight=0):
	"""Align g and p.
	- sub: a dictionary of substitution probabilities
	- ins: a dictionary of insertion probabilities
	- dlt: a dictionary of deletion probabilities
	- gcd: grapheme category dictionary
	- pcd: phoneme category dictionary
	"""
	if not g[0] == '#': g.insert(0, '#')
	if not p[0] == '#': p.insert(0, '#')
	lg = len(g); lp = len(p)
	M = [[[0, None]]*lp for i in range(lg)]
	for i in range(1, lg): M[i][0] = [-50*i, (i-1, 0)]
	for j in range(1, lp): M[0][j] = [-50*j, (0, j-1)]
	for i in range(1, lg):
		for j in range(1, lp):
			sc_s = M[i-1][j-1][0]
			if set(gcd[g[i]]).intersection(set(pcd[p[j]])) == set([]): sc_s -= 100
			try: sc_s += sub[g[i]][p[j]]
			except KeyError: pass
			sc_i = M[i][j-1][0] - 1
			if M[i][j-1][1] == (i, j-2): sc_i -= 50
			try: sc_i += ins[g[i-1]][p[j]] * ins_weight
			except KeyError: pass
			sc_d = M[i-1][j][0] - 1
			if M[i-1][j][1] == (i-2, j): sc_d -= 50
			try: sc_d += dlt[g[i]][p[j-1]] * dlt_weight
			except KeyError: pass
			M[i][j] = max([[sc_s, (i-1, j-1)], [sc_i, (i, j-1)], [sc_d, (i-1, j)]])
	#print(M)
	crumbs = [(lg-1, lp-1)]
	while crumbs[-1] is not None:
		i, j = crumbs[-1]
		crumbs.append(M[i][j][1])
	crumbs.reverse()
	crumbs = crumbs[1:]
	ga = [g[0]]; pa = [p[0]]
	for i in range(1, len(crumbs)):
		pi, pj = crumbs[i-1]
		ni, nj = crumbs[i]
		if pi == ni: ga.append('_')
		else: ga.append(g[ni])
		if pj == nj: pa.append('_')
		else: pa.append(p[nj])
	return ga, pa

def update_count(ga, pa, sub, ins, dlt):
	"""Update edit counts."""
	for i in range(1, len(ga)):
		if ga[i] != '_' and pa[i] != '_':
			if not ga[i] in sub: sub[ga[i]] = {}
			if not pa[i] in sub[ga[i]]: sub[ga[i]][pa[i]] = 0
			sub[ga[i]][pa[i]] += 1
		elif ga[i] == '_' and pa[i] != '_': 
			if not ga[i-1] in ins: ins[ga[i-1]] = {}
			if not pa[i] in ins[ga[i-1]]: ins[ga[i-1]][pa[i]] = 0
			ins[ga[i-1]][pa[i]] += 1
		elif ga[i] != '_' and pa[i] == '_':
			if not ga[i] in dlt: dlt[ga[i]] = {}
			if not pa[i-1] in dlt[ga[i]]: dlt[ga[i]][pa[i-1]] = 0
			dlt[ga[i]][pa[i-1]] += 1
	return sub, ins, dlt


def initialize(g, p, sub):
	"""Initialize substitution dictionary with g-p pairs of equal lengths."""
	if len(g) == len(p):
		for i in range(len(g)):
			if not g[i] in sub: sub[g[i]] = {}
			if not p[i] in sub[g[i]]: sub[g[i]][p[i]] = 0
			sub[g[i]][p[i]] += 1
	return sub

def filter(sub, xcd, ycd):
	"""Filter substitution dictionary to prevent non-sensical matching.
	Args:
	- sub: substitution dictionary, sub[x] = [... y ...]
	- xcd: category dictionary for x, xcd[x] = category
	- ycd: category dictionary for y, ycd[y] = category
	"""
	for x in sub:
		sub[x] = {y:sub[x][y] for y in sub[x] if set(xcd[x]).intersection(set(ycd[y])) != set([])}
	return sub

def load_categories(vf):
	"""Load category file."""
	cd = {}
	for line in open(vf):
		x, c = line.strip().split('\t')
		cd[x] = c.split(',')
	return cd

def normalize(cd):
	"""Normalize frequencies."""
	for x in cd:
		total = sum(cd[x].values())
		cd[x] = {y:cd[x][y]/total for y in cd[x]}
	return cd

def epoch(gl, pl, sub, ins, dlt, gcd, pcd, ins_weight=0, dlt_weight=0):
	"""Run one epoch."""
	n_sub = {}; n_ins = {}; n_dlt = {}
	for n in range(len(gl)):
		ga, pa = align(gl[n], pl[n], sub, ins, dlt, gcd, pcd, ins_weight, dlt_weight)
		update_count(ga, pa, n_sub, n_ins, n_dlt)
	return normalize(filter(n_sub, gcd, pcd)), normalize(n_ins), normalize(n_dlt)
	

if __name__ == '__main__':
	gl = [line.strip().split() for line in open('./data/cmudict.train.g.txt')]
	pl = [line.strip().split() for line in open('./data/cmudict.train.p.txt')]
	gcd = load_categories('./data/vocab.g'); pcd = load_categories('./data/vocab.p')
	ins_weight = 0.1; dlt_weight = 0.1
	if len(sys.argv) > 1:
		ins = {}; dlt = {}; sub = {}
		for i in range(len(gl)): initialize(gl[i], pl[i], sub)
		sub = normalize( filter(sub, gcd, pcd) )
		for e in range(10):
			sys.stderr.write('# Epoch '+str(e+1)+'...'+' '*5+'\r')
			sub, ins, dlt = epoch(gl, pl, sub, ins, dlt, gcd, pcd, ins_weight, dlt_weight)
			print(sub)
		sys.stderr.write('\n# Training complete.\n')
		pickle.dump(sub, open('sub.pickle', 'wb'))
		pickle.dump(ins, open('ins.pickle', 'wb'))
		pickle.dump(dlt, open('dlt.pickle', 'wb'))
	else:
		sub = pickle.load(open('sub.pickle', 'rb'))
		ins = pickle.load(open('ins.pickle', 'rb'))
		dlt = pickle.load(open('dlt.pickle', 'rb'))
		fg = open('cmudict.train.g.aligned.txt', 'w')
		fp = open('cmudict.train.p.aligned.txt', 'w')
		for n in range(len(gl)):
			ga, pa = align(gl[n], pl[n], sub, ins, dlt, gcd, pcd, ins_weight, dlt_weight)
			fg.write(' '.join(ga)+'\n')
			fp.write(' '.join(pa)+'\n')
		fg.close(); fp.close()