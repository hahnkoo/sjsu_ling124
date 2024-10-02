def align(x, y):
	x = ['#'] + x; y = ['#'] + y
	m = len(x); n = len(y)
	C = [[[0, None]] * n for i in range(m)]
	for j in range(1, n): C[0][j] = [C[0][j-1][0]+1, (0, j-1)]
	for i in range(1, m): C[i][0] = [C[i-1][0][0]+1, (i-1, 0)]
	for i in range(1, m):
		for j in range(1, n):
			C[i][j] = min([[C[i-1][j-1][0]+int(x[i]!=y[j]), (i-1, j-1)],
						   [C[i-1][j][0]+1, (i-1, j)],
						   [C[i][j-1][0]+1, (i, j-1)]])
	crumbs = [(m-1, n-1)]
	while crumbs[-1] != (0, 0):
		i, j = crumbs[-1]
		crumbs.append(C[i][j][1])
	crumbs.reverse()
	xa = []; ya = []
	for t in range(1, len(crumbs)):
		pi, pj = crumbs[t-1]
		i, j = crumbs[t]
		if pi == i: xa.append('__')
		elif pi != i: xa.append(x[i])
		if pj == j: ya.append('__')
		elif pj != j: ya.append(y[j])
	return xa, ya

def stats(x, y):
	xa, ya = align(x, y)
	ne = 0; n = len(xa)
	for i in range(n): ne += int(xa[i] != ya[i])
	return ne, n

if __name__ == '__main__':
	x = 'N AO'.split()
	y = 'N AO L T'.split()
	print( stats(x, y) )