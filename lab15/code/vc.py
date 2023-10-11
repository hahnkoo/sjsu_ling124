import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class model(nn.Module):

	def __init__(self):
		super(model, self).__init__()
		self.layers = nn.Sequential(
						nn.Linear(2, 4),
						nn.Sigmoid(),
						nn.Linear(4, 3)
						)
		self.prob = nn.Softmax(dim=-1)

	def forward(self, x):
		return self.layers(x)

	def update(self, x, y, alpha):
		self.zero_grad()
		optimizer = optim.Adam(self.parameters(), lr=alpha)
		loss_function = nn.CrossEntropyLoss()
		y_ = self.forward(x)
		loss = loss_function(y_, y)
		loss.backward()
		optimizer.step()
		return loss 


def batch(f, size, y2i):
	xs = []; ys = []
	done = False; i = 0
	while not done:
		line = f.readline()
		if line == '': done = True
		else:
			ll = line.strip().split(',')
			x = np.array(ll[1:]).astype(float)
			xs.append(x)
			y = y2i[ll[0]]
			ys.append(y)
		i += 1
		if i >= size: done = True
	xs = torch.Tensor(xs)
	ys = torch.LongTensor(ys)
	return xs, ys

def onehot(ys, cl):
	out = []
	n = len(cl)
	for y in ys:
		yoh = np.zeros(n)
		yoh[cl.index(y)] = 1
		out.append(yoh)
	return out

def load_classes(fn):
	f = open(fn)
	y2i = {}
	for line in f: y2i[line.strip()] = len(y2i)
	f.close()
	return y2i

if __name__ == '__main__':
	y2i = {'aa':0, 'iy':1, 'uw':2}
	m = model()

	# Train
	n_epochs = 1000; bs = 128; alpha = 0.001
	trf = '../data/aiu.train'
	for epoch in range(n_epochs):
		f = open(trf)
		eof = False
		bi = 1
		while not eof:
			xs, ys = batch(f, bs, y2i)
			if len(ys) == 0: eof = True
			else:
				loss = m.update(xs, ys, alpha)
				print(loss.item())
				bi += 1
		f.close()

	# Test
	tsf = '../data/aiu.test'
	f = open(tsf)
	eof = False
	c = 0.0; n = 0.0
	while not eof:
		xs, ys = batch(f, 1, y2i)
		if len(ys) == 0: eof = True
		else:
			ys_ = m.forward(xs)
			values, indices = ys_.max(1)
			print (xs, ys)
			c += float((indices == ys).sum())
			n += len(ys)
	f.close()
	print ('Acc: '+str(c/n*100)+' %')
