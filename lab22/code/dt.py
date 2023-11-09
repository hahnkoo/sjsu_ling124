import sys, pickle, re
import numpy as np
from sklearn import tree
#from sklearn.tree.export import export_text

def load(fn):
	X = []; Y = []
	f = open(fn)
	for line in f:
		ll = line.strip().split('\t')
		X.append(ll[1].split()); Y.append(ll[0])
	f.close()
	return X, Y

def vocab(X, Y):
	xv = {}; yv = {}
	for x in X:
		for xi in x: xv[xi] = 1
	for y in Y:
		yv[y] = 1
	xv = {x:i for i, x in enumerate(sorted(xv.keys()))}
	yv = {y:i for i, y in enumerate(sorted(yv.keys()))}
	return xv, yv

def to_array(X, Y, xv, yv):
	Xa = []
	for x in X:
		xa = np.zeros(0)
		for xi in x:
			xia = np.zeros(len(xv))
			xia[xv[xi]] = 1
			xa = np.concatenate((xa, xia))
		Xa.append(xa)
	Ya = [yv[y] for y in Y]
	return np.array(Xa), np.array(Ya)
		
def transcribe(clf, g, g2i, i2p, n=3):
	if not g[0] == '#': g.insert(0, '#')
	g = ['#']*(n-1) + g + ['#']*n
	X = []
	for i in range(n, len(g)-n):
		gi = g[i-n:i+n+1]
		x = np.zeros(0)
		for gij in gi:
			xj = np.zeros(len(g2i))
			xj[g2i[gij]] = 1
			x = np.concatenate((x, xj))
		X.append(x)
	X = np.array(X)
	Y = clf.predict(X)
	p = ' '.join([i2p[y] for y in Y])
	p = re.sub('_', '', p)
	p = re.sub(' {2,}', ' ', p)
	return p.strip()
		
	
if __name__ == '__main__':
	n = int(sys.argv[1])
	mode = sys.argv[2] # 'train' or 'test'
	fn = sys.argv[3]
	if mode == 'train':
		X, Y = load(fn)
		x2i, y2i = vocab(X, Y)
		X, Y = to_array(X, Y, x2i, y2i)
		clf = tree.DecisionTreeClassifier()
		clf = clf.fit(X, Y)
		r = tree.export_text(clf)
		Y_ = clf.predict(X)
		nc = 0
		for i in range(len(Y_)):
			if Y_[i] == Y[i]: nc += 1
		print('acc:', nc/len(Y_))
		pickle.dump(clf, open('g2p.'+str(n)+'.pickle', 'wb'))
		pickle.dump(x2i, open('vocab.g.'+str(n)+'.pickle', 'wb'))
		pickle.dump(y2i, open('vocab.p.'+str(n)+'.pickle', 'wb'))
	elif mode == 'test':
		clf = pickle.load(open('g2p.'+str(n)+'.pickle', 'rb'))
		x2i = pickle.load(open('vocab.g.'+str(n)+'.pickle', 'rb'))
		y2i = pickle.load(open('vocab.p.'+str(n)+'.pickle', 'rb'))
		i2y = {y2i[y]:y for y in y2i}
		pfn = re.sub('\.g\.', '.p.', fn)
		pf = open(pfn)
		nc = 0; nt = 0
		for line in open(fn):
			g = line.strip().split()
			p_ = transcribe(clf, g, x2i, i2y, n)
			p = pf.readline().strip()
			print(p_==p, p_, '\t', p)
			if p_ == p: nc += 1
			nt += 1
		print('## ACC:', nc, '/', nt, '=', nc/nt)