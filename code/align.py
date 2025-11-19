import sys

def score(x, y):
	d = [[(i, (i-1, 0))] for i in range(len(x))]
	for j in range(1, len(y)): d[0].append((j, (0, j-1)))
	for i in range(1, len(x)):
		for j in range(1, len(y)):
			sub = d[i-1][j-1][0] + int(x[i] != y[j])
			ins = d[i-1][j][0] + 1
			dlt = d[i][j-1][0] + 1 
			cand = [(sub, (i-1, j-1)), (ins, (i-1, j)), (dlt, (i, j-1))]
			d[i].append(min(cand))
	return d	

def trace(x, y, d):
	coord = (len(x)-1, len(y)-1)
	out = [coord]
	while coord != (0, 0):
		coord = d[coord[0]][coord[1]][-1]
		out.append(coord)
	out.reverse()
	return out
		
def pretty(x, y, a):
	xa = ['#']; ya = ['#']
	for n in range(1, len(a)):
		pi, pj = a[n-1]
		i, j = a[n]
		if i == pi: xa.append('__')
		else: xa.append(x[i])
		if j == pj: ya.append('__')
		else: ya.append(y[j])
	return xa, ya

def count_errors(ref, hyp):
	r = ['#'] + ref.strip().split()
	h = ['#'] + hyp.strip().split()
	a = trace(r, h, score(r, h))
	ra, ha = pretty(r, h, a)
	ne = 0; n = len(ra)
	for i in range(n):
		if ra[i] != ha[i]: ne += 1
	return ne, n

def wer(reflist, hyplist):
	ne = 0; n = 0
	for i in range(len(reflist)):
		nei, ni = count_errors(reflist[i], hyplist[i])
		ne += nei; n += ni
	return ne, n, ne/n

if __name__ == '__main__':
	rl = open(sys.argv[1]).readlines()
	hl = open(sys.argv[2]).readlines()
	for rli, hli in zip(rl, hl):
		r = ['#'] + rli.strip().split()
		h = ['#'] + hli.strip().split()
		d = score(r, h)
		a = trace(r, h, d)
		ra, ha = pretty(r, h, a)
		print ()
		print ( '  '.join(ra[1:]))
		print ('  '.join(ha[1:]))
	print ()
