import sys, random

if __name__ == '__main__':
	fn = sys.argv[1]
	f = open(fn)
	lines = f.readlines()
	f.close()
	random.shuffle(lines)
	n = int(0.9*len(lines))
	trf = open('bullshit.train.txt', 'w')
	for line in lines[:n]: trf.write(line)
	trf.close()
	tsf = open('bullshit.test.txt', 'w')
	for line in lines[n:]: tsf.write(line)
	tsf.close()