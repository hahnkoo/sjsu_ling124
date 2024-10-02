"""Yarowsky's decision list for homograph disambiguation for LING 124"""

__author__ = """Hahn Koo (hahn.koo@sjsu.edu)"""


import sys, re, math, pickle, argparse 


def feature_unit(unit):
	"""Get features from the unit."""
	out = {} 
	ll = unit.split('/')
	if len(ll) == 3: out = {'word':ll[0], 'lemma':ll[1], 'pos':ll[2]}
	return out

def feature_context(lemma, k, line): 
	"""Rewrite line in terms of contextual features."""
	feats = []
	ll = line.strip().split('\t')
	labels = []
	sent = ll[-1]
	if len(ll) == 2: labels = ll[0].split()
	words = sent.split()
	tis = []
	for i in range(len(words)):
		feat = feature_unit(words[i])
		if feat.get('lemma') == lemma: tis.append(i)
	for ti in tis:
		feat = []
		# to the left
		try:
			left = feature_unit(words[ti-1])
			feat.append('-1W:'+left['word'])
			feat.append('-1L:'+left['lemma'])
			feat.append('-1P:'+left['pos'])
		except IndexError: pass 
		except KeyError: pass
		# to the right
		try:
			right = feature_unit(words[ti+1])
			feat.append('+1W:'+right['word'])
			feat.append('+1L:'+right['lemma'])
			feat.append('+1P:'+right['pos'])
		except IndexError: pass 
		except KeyError: pass
		# within k words
		for i in range(max(0,ti-k), min(ti+k,len(words))):
			try:
				there = feature_unit(words[i])
				feat.append('inkW:'+there['word'])
				feat.append('inkL:'+there['lemma'])
				feat.append('inkP:'+there['pos'])
			except KeyError: pass
		feat = set(feat)
		feats.append(feat)
	if len(labels) == 1 and len(labels) < len(feats): labels = labels*len(feats)
	return labels, feats 

def update_count(lemma, k, delta, label_set, line, fd):
	"""Update feat,label co-occurrence frequencies."""
	labels, feats = feature_context(lemma, k, line)
	for i in range(len(feats)):
		label = labels[i]
		feat = feats[i]
		for av in feat:
			if not av in fd:
				fd[av] = {}
				for c in label_set: fd[av][c] = delta
			fd[av][label] += 1
	return fd

def log_odds(fd):
	"""List abs(log-odds) of feat,label co-occurence."""
	out = []
	for feat in fd:
		t = sum(fd[feat].values())
		temp = []
		for label in fd[feat]:
			p = fd[feat][label] / t
			temp.append((p, label))
		temp.sort(reverse=True)
		pred = temp[0][1]
		prob = temp[0][0]
		score = abs(math.log(prob/(1-prob)))
		out.append((score, feat, pred))
	return out

def train(lemma, k, delta, label_set, fn):
	"""Train a decision list using a file named fn."""
	fd = {}
	f = open(fn, errors='replace')
	for line in f:
		fd = update_count(lemma, k, delta, label_set, line, fd)
	f.close()
	#pickle.dump(fd, open('lead.freq', 'w'))
	dl = log_odds(fd)
	dl.sort(reverse=True)
	return dl 

def apply(lemma, k, dl, feat):
	"""Apply decision list to classify lemma in line."""
	prediction = None
	for rule in dl:
		if rule[1] in feat:
			prediction = rule[-1]
			break
	return prediction



if __name__ == '__main__':

	parse = argparse.ArgumentParser()
	parse.add_argument('--classes', dest='classes', action='store', default='liyd,lehd')
	parse.add_argument('--lemma', dest='target_lemma', action='store', default='lead')
	parse.add_argument('--k', dest='k', action='store', type=int, default=5)
	parse.add_argument('--delta', dest='delta', action='store', type=float, default=0.01)
	parse.add_argument('--train', dest='trf', action='store', default=None)
	parse.add_argument('--test', dest='tsf', action='store', default=None)
	parse.add_argument('--save', dest='svf', action='store', default=None) 
	parse.add_argument('--load', dest='ldf', action='store', default=None) 
	parse.add_argument('--show', dest='show', action='store_true', default=False)
	parse.add_argument('--acc', dest='acc', action='store_true', default=False)
	a = parse.parse_args()

	dl = []
	if a.trf:
		dl = train(a.target_lemma, a.k, a.delta, a.classes.split(','), a.trf)
		sys.stderr.write('# Decision list trained on '+a.trf+'.\n')
	elif a.ldf:
		sys.stderr.write('# Decision list loaded from '+a.ldf+'\n')
		dlf = open(a.ldf, 'rb')
		dl = pickle.load(dlf)
		dlf.close()

	if dl == []:
		sys.stderr.write('# The decision list is empty. Either train one from scratch or load a pre-trained list.\n')
		sys.exit(2)
	else:
		if a.tsf:
			sys.stderr.write('# Testing the decision list on '+a.tsf+'\n')
			nc = 0.0; nt = 0.0
			f = open(a.tsf, errors='replace')
			for line in f:
				ll = line.strip().split('\t')
				sent = ll[-1]
				labels, feats = feature_context(a.target_lemma, a.k, line)
				for i in range(len(feats)):
					c_hat = apply(a.target_lemma, a.k, dl, feats[i])
					print (c_hat + '\t' + sent)
				if len(ll) == 2 and a.acc:
					true = ll[0]
					if true == c_hat: nc += 1
				nt += 1
			f.close()
			if a.acc: sys.stderr.write('\n# Accuracy on '+str(a.tsf)+': '+str(nc/nt)+'\n')
		if a.show:
			sys.stderr.write('# Showing rules in the decision list:\n')
			print ('Score\tLabel\tFeature (attrib:value)')
			for rule in dl:
				print ('%.4f' % rule[0], '\t', rule[2], '\t', rule[1]) 
		if a.svf:
			dlf = open(a.svf, 'wb')
			pickle.dump(dl, dlf)
			dlf.close()
			sys.stderr.write('# Decision list saved in '+a.svf+'.\n')
