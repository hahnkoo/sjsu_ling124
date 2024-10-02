"""Generate training data."""

import sys

def segment(ga, pa):
	"""Segment alignment pair."""
	out = []; g = []; p = []
	i = 0
	while i < len(ga):
		if ga[i] == '_':
			p.append(pa[i])
		else:
			if g != []: out.append((g[0], p))
			g = [ga[i]]; p = [pa[i]]
		i += 1
	out.append((g[0], p))
	return out

def window(example, n=3):
	"""Get windowed feature examples from (segmented) example."""
	out = []
	example = [('#', ('#',))]*(n-1) + example + [('#', ('#',))]*n
	for i in range(n, len(example)-n):
		gc = [g for g, p in example[i-n:i+n+1]]
		p = example[i][1]
		out.append((p, gc))
	return out
		

if __name__ == '__main__':
	n = int(sys.argv[1])
	gal = [line.strip().split() for line in open('cmudict.train.g.aligned.txt')]
	#gal = ['# A C C E P T A N C E'.strip().split()]
	pal = [line.strip().split() for line in open('cmudict.train.p.aligned.txt')]
	#pal = ['# AE K S EH P T AH N S _'.strip().split()]
	for i in range(len(gal)):
		parts = segment(gal[i], pal[i])
		#for part in parts: print(part)
		#print()
		for point in window(parts, n):
			print(' '.join(point[0]) + '\t' + ' '.join(point[1]))