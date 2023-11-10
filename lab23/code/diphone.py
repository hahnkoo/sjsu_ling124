import re, glob
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal.windows as windows


def ref_diphones(dic_fn, wav_path):
	d = {}
	f = open(dic_fn)
	for i in range(4): trash = f.readline()
	for line in f:
		dp, fn, b, m, e = line.strip().split()
		d[dp] = [wav_path+'/'+fn+'.wav', (float(b), float(e))]
	f.close()
	return d

def load_lab(fn, dpd, ref):
	hit = False
	diphone = []; boundary = []; phones = []
	for line in open(fn):
		if hit:
			t, n, p = line.strip().split()
			phones.append((p, float(t)))
		if line.strip() == '#': hit = True
	phones.insert(0, ('#', 0))
	biphones = []
	for i in range(1, len(phones)-1):
		bip = phones[i][0]+'-'+phones[i+1][0]
		biphones.append([bip, [phones[t][1] for t in range(i-1, i+2)]])
	fn = re.sub('\.lab$', '.wav', fn)
	for bip, [bt, mt, et] in biphones:
		if bip in ref:
			bip_fn, (bip_b, bip_e) = ref[bip]
			if bip_fn[-12:] == fn[-12:]: 
				if mt >= bip_b and mt <= bip_e:
					dpd[bip] = [fn, [bt, mt, et]]
	return dpd

def midpoint(d):
	md = {}
	for bp in d:
		p1, p2 = bp.split('-')
		fn, [b, m, e] = d[bp]
		bm = (b+m)/2; em = (m+e)/2
		#if p1 in ['p', 't', 'k', 'b', 'd', 'g']: bm = b+0.3*(m-b)
		#if p2 in ['p', 't', 'k', 'b', 'd', 'g']: em = e-0.3*(e-m)
		md[bp] = [fn, [bm, em]]
	return md

def extract(wfn, begin, end):
	Fs, x = wavfile.read(wfn)
	b = int(Fs*begin); e = int(Fs*end)
	return x[b:e]

def transcribe_diphone(x):
	if type(x) == str: x = x.strip().split()
	x = ['pau']+x+['pau']
	return ['-'.join(x[i:i+2]) for i in range(len(x)-1)]

def concatenate(dt, dpd, taper=False):
	out = np.zeros(0)
	for d in dt:
		w = extract(dpd[d][0], dpd[d][1][0], dpd[d][1][-1])
		if taper: w = windows.tukey(len(w)) * w
		out = np.concatenate((out, w))
	return out


if __name__ == '__main__':
	path = '../data/'
	ref = ref_diphones(path+'/dic/kaldiph.est', path+'/wav')
	dpd = {}
	for lfn in glob.glob(path+'lab/*'): dpd = load_lab(lfn, dpd, ref)
	md = midpoint(dpd)
	for key in md:
		print(key, md[key][1][0], md[key][1][-1], 'vs', ref[key][1][0], ref[key][1][-1])
	
