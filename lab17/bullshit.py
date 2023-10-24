"""RNN Bullshit Generator"""

__author__ = """Hahn Koo (hahn.koo@sjsu.edu)"""

import sys, argparse, random
import torch
import torch.nn as nn
import torch.optim as optim

class model(nn.Module):

	def __init__(self, vs, ed, hs, rnn_type):
		"""
		Args:
		- vs: vocab size
		- ed: embedding dimension
		- hs: hidden size
		- rnn_type: 'RNN', 'GRU'
		"""
		super(model, self).__init__()
		self.embedding = nn.Embedding(vs, ed)
		self.hs = hs
		if rnn_type == 'RNN': self.rnn = nn.RNN(ed, hs)
		elif rnn_type == 'GRU': self.rnn = nn.GRU(ed, hs)
		self.fc = nn.Sequential( nn.Linear(hs, vs), nn.LogSoftmax(dim=-1) )

	def forward(self, x):
		x = self.embedding(x)
		s = x.view(x.shape[0], 1, x.shape[-1])
		o, h = self.rnn(s)
		y_ = self.fc(o)
		return y_

	def train(self, x, alpha):
		lf = nn.NLLLoss()
		self.zero_grad()
		o = optim.SGD(self.parameters(), lr=alpha)
		y_ = self.forward(x[:-1]).view(x.shape[0]-1, -1)
		y = x[1:]
		loss = lf(y_, y)
		loss.backward()
		o.step()
		return loss

	def sample(self, w2i, i2w, maxlen=50):
		stop = False
		h = torch.zeros(1, 1, self.hs)
		x = torch.tensor([w2i['<s>']])
		out = []
		while not stop:
			s = self.embedding(x).view(1, 1, -1)
			o, h = self.rnn(s, h)
			y_ = self.fc(o).view(-1)
			i = torch.multinomial(torch.exp(y_), 1).item()
			w = i2w[i]
			if w != '<UNK>': out.append(w)
			if w == '</s>': stop = True
			if len(out) >= maxlen: stop = True
			x = torch.tensor([w2i[w]])
		return out

	def evaluate(self, x):
		y_ = self.forward(x[:-1]).view(x.shape[0]-1, -1)
		lp = 0.0
		for i in range(1, len(x)): lp += y_[i-1, x[i]]
		return lp


def create_vocab(fn):
	w2i = {'<s>':0, '</s>':1, '<UNK>':2}
	f = open(fn)
	for line in f:
		for w in line.strip().split():
			if not w in w2i: w2i[w] = len(w2i)
	f.close()
	i2w = {w2i[w]:w for w in w2i}
	return w2i, i2w

def load_vocab(fn):
	w2i = {}
	f = open(fn)
	for line in f:
		if line.strip() != '': w2i[line.strip()] = len(w2i)
	f.close()
	i2w = {w2i[w]:w for w in w2i}
	return w2i, i2w

def index_words(words, w2i, pad=False, add_noise=False):
	out = [w2i.get(w, w2i['<UNK>']) for w in words]
	if pad: out = [w2i['<s>']] + out + [w2i['</s>']]
	if add_noise:
		ni = random.randint(0, len(out)*5)
		if ni > 0 and ni < len(out)-1: out[ni] = w2i['<UNK>']
	return torch.tensor(out)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--vocab', dest='vf', type=str)
	parser.add_argument('--train', dest='trf', type=str)
	parser.add_argument('--alpha', dest='alpha', type=float, default=0.01)
	parser.add_argument('--epochs', dest='epochs', type=int, default=0)
	parser.add_argument('--save', dest='save', type=str)
	parser.add_argument('--load', dest='load', type=str)
	parser.add_argument('--embedding_dim', dest='ed', type=int, default=100)
	parser.add_argument('--hidden_size', dest='hs', type=int, default=100)
	parser.add_argument('--rnn', dest='rnn', type=str, default='GRU')
	parser.add_argument('--data_size', dest='data_size', type=int, default=5000)
	parser.add_argument('--test', dest='tsf', type=str)
	parser.add_argument('--sample', dest='sample', type=int)
	args = parser.parse_args()
	w2i = {}; i2w = {}
	if args.vf: w2i, i2w = load_vocab(args.vf)
	elif args.trf:
		w2i, i2w = create_vocab(args.trf)
		f = open('bullshit.vocab.txt', 'w')
		for i in sorted(i2w): f.write(i2w[i]+'\n')
		f.close()

	m = model(len(w2i), args.ed, args.hs, args.rnn)
	if args.load: m.load_state_dict( torch.load(args.load) )
	if args.trf:
		f = open(args.trf)
		lines = f.readlines()
		f.close()
		for epoch in range(args.epochs):
			random.shuffle(lines)
			total_loss = 0; i = 0
			for line in lines[:args.data_size]:
				x = index_words(line.strip().split(), w2i, pad=True, add_noise=True)
				loss = m.train(x, args.alpha)
				total_loss += loss.item()
				i += 1
				sys.stderr.write('## Sentence '+str(i)+' loss: '+str(loss.item())+' '*5+'\r')
			sys.stderr.write('\n# Epoch '+str(epoch+1)+' total loss: '+str(total_loss)+'\n')
			f.close()
		if args.save: torch.save(m.state_dict(), args.save)

	if args.tsf:
		f = open(args.tsf)
		tlp = 0.0
		for line in f:
			x = index_words(line.strip().split(), w2i, pad=True)
			tlp += m.evaluate(x).item()
		f.close()
		print('# Sum of log probabilities:', tlp)
	
	if args.sample:
		for i in range(args.sample): print( ' '.join(m.sample(w2i, i2w)) )
	