"""Code for Unit Selection Synthesis"""

def join(x1, x2):
	score = 0
	if x1['diphone'] == '<s>' or x2['diphone'] == '</s>': return score
	if x1['diphone'] == x2['previous']: score += 1
	if x1['source'] == x2['source']: score += 3
	if x1['end'] == x2['start']: score += 5
	return score
	
def target(x, y):
	score = 0
	if x['diphone'] == '</s>': return score
	if x['word'] == y['word']: score += 5
	score -= abs(float(x['duration_z']) - float(y['duration_z']))
	score -= abs(float(x['pitch_z']) - float(y['pitch_z']))
	score -= abs(float(x['rms_z']) - float(y['rms_z']))
	return score

def trellis(target_list, d2f, sd):
	t = [{'<s>':[0, None]}]
	for y in target_list:
		col = {fn:[-1e+100, None] for fn in d2f[y['diphone']]}
		for x2 in col:
			for x1 in t[-1]:
				score = t[-1][x1][0]
				score += join(sd[x1], sd[x2])
				score += target(sd[x2], y)
				if score > col[x2][0]: col[x2] = [score, x1]
		t.append(col)
	return t

def pretty(t):
	for i in range(len(t)):
		print('\n## Column '+str(i+1))
		for fn in t[i]:
			print('# '+fn+': delta='+str(t[i][fn][0])+', crumb='+str(t[i][fn][1]))

def load_spec(csvfn):
	f = open(csvfn)
	header = f.readline().strip()
	features = header.split(',')[1:]
	d = {'<s>':{'diphone':'<s>'}, '</s>':{'diphone':'</s>'}}
	for line in f:
		ll = line.strip().split(',')
		d[ll[0]] = {features[i-1]:ll[i] for i in range(1, len(ll))}
	f.close()
	return d

def diphone2fn(d):
	d2f = {}
	for fn in d:
		if not d[fn]['diphone'] in d2f: d2f[d[fn]['diphone']] = []
		d2f[d[fn]['diphone']].append(fn)
	return d2f

if __name__ == '__main__':
	sd = load_spec('sea_diphones_spec.csv')
	d2f = diphone2fn(sd)
	target_list = [{'diphone':'pau-s', 'word':'sea', 'duration_z':0, 'pitch_z':0, 'rms_z':0}, {'diphone':'s-iy', 'word':'sea', 'duration_z':1, 'pitch_z':1, 'rms_z':1}, {'diphone':'iy-pau', 'word':'sea', 'duration_z':0, 'pitch_z':0, 'rms_z':0}, {'diphone':'</s>'}]
	t = trellis(target_list, d2f, sd)
	pretty(t)
	
