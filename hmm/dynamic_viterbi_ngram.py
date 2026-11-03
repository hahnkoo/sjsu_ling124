"""Dynamic Viterbi continuous ASR with optional word N-gram language models.

The HMMs and lexicon follow the LING 111 Day 3 notebook conventions.
All HMM entry/exit states are non-emitting. A word must emit >= 1 frame.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log
import numpy as np


class NgramLM:
	"""Add-k smoothed word N-gram model, order 1..3.

	Fit from one whitespace-tokenized sentence per line.
	Probabilities are normalized over the supplied recognition vocabulary plus </s>.
	"""
	def __init__(self, vocabulary, order=2, alpha=0.1):
		if order not in (1, 2, 3):
			raise ValueError('order must be 1, 2, or 3')
		if alpha <= 0:
			raise ValueError('alpha must be positive')
		self.vocabulary = tuple(dict.fromkeys(vocabulary))
		self.vocab_set = set(self.vocabulary)
		self.order = order
		self.alpha = float(alpha)
		self.counts = Counter()
		self.context_counts = Counter()
		self.start = ('<s>',) * (order - 1)
		self.fitted = False

	def fit(self, sentences):
		self.counts.clear()
		self.context_counts.clear()
		for sentence in sentences:
			words = sentence.split() if isinstance(sentence, str) else list(sentence)
			if any(w not in self.vocab_set for w in words):
				unknown = sorted(set(words) - self.vocab_set)
				raise ValueError(f'Training words missing from lexicon: {unknown[:10]}')
			context = self.start
			for word in words + ['</s>']:
				self.counts[(context, word)] += 1
				self.context_counts[context] += 1
				context = self.advance(context, word)
		self.fitted = True
		return self

	@classmethod
	def from_file(cls, path, vocabulary, order=2, alpha=0.1):
		with open(path, encoding='utf-8-sig') as f:
			return cls(vocabulary, order, alpha).fit(line.strip() for line in f if line.strip())

	def advance(self, history, word):
		if self.order == 1:
			return ()
		return (history + (word,))[-(self.order - 1):]

	def logprob(self, word, history):
		if not self.fitted:
			raise RuntimeError('Call fit() before decoding')
		if word not in self.vocab_set and word != '</s>':
			return -np.inf
		if len(history) != self.order - 1:
			raise ValueError('Incorrect history length')
		numerator = self.counts[(history, word)] + self.alpha
		denominator = self.context_counts[history] + self.alpha * (len(self.vocabulary) + 1)
		return log(numerator / denominator)


@dataclass(frozen=True)
class Edge:
	dest: int
	logp: float


class WordLoopNetwork:
	"""Compile word pronunciations to word-specific copies of phone emitting states."""
	def __init__(self, models, lexicon, word_penalty=0.0):
		self.models = models
		self.word_penalty = float(word_penalty)
		self.words = []
		self.labels = []
		self.emitting = []
		self.edges = []
		self.starts = []  # (state, log entry probability)
		self.ends = []    # (state, log exit probability)
		self.start_word = {}
		self.end_word = {}
		self.exit_logprob = {}
		for word, phones in lexicon.items():
			if not phones or any(phone not in models for phone in phones):
				continue
			self.words.append(word)
			previous_exits = None
			for occurrence, phone in enumerate(phones):
				model = models[phone]
				n = len(model.states)
				if n < 3:
					raise ValueError(f'{phone}: expected entry, emitting, exit states')
				base = len(self.labels)
				for j in range(1, n-1):
					self.labels.append((word, phone, occurrence, j))
					self.emitting.append((phone, j))
					self.edges.append([])
				entry = [(base+j-1, log(model.trans[0,j])) for j in range(1,n-1)
						 if model.trans[0,j] > 0]
				exits = [(base+j-1, log(model.trans[j,n-1])) for j in range(1,n-1)
						 if model.trans[j,n-1] > 0]
				for j in range(1,n-1):
					for k in range(1,n-1):
						if model.trans[j,k] > 0:
							self.edges[base+j-1].append(
								Edge(base+k-1, log(model.trans[j,k])))
				if previous_exits is None:
					for dest, lp in entry:
						self.starts.append((dest, lp))
						self.start_word[dest] = word
				else:
					for prev, exit_lp in previous_exits:
						for dest, entry_lp in entry:
								self.edges[prev].append(Edge(dest, exit_lp+entry_lp))
				previous_exits = exits
			for state, lp in previous_exits:
				self.ends.append((state, lp))
				self.end_word[state] = word
				self.exit_logprob[state] = lp
		if not self.words:
			raise ValueError('No lexicon entries match the phone inventory')
		self.n_states = len(self.labels)


class DynamicTrellis:
	"""Sparse Viterbi columns keyed by (acoustic_state, LM_history).

	Each entry is (log_score, previous_key, begins_word).
	The LM history contains words preceding the current word, updated when
	a new word begins. This prevents invalid merging of different histories.
	"""
	def __init__(self, network, lm=None, lm_weight=1.0):
		self.network = network
		self.lm = lm
		self.lm_weight = float(lm_weight)
		if lm is not None and set(network.words) != lm.vocab_set:
			raise ValueError('LM vocabulary must match the network vocabulary')
		self.columns = []
		self.stats = []

	def _score_emission(self, state, frame, cache):
		phone_state = self.network.emitting[state]
		if phone_state not in cache:
			phone, j = phone_state
			cache[phone_state] = float(self.network.models[phone].states[j].logpdf(frame))
		return cache[phone_state]

	@staticmethod
	def _offer(col, key, score, previous, boundary):
		if np.isfinite(score) and (key not in col or score > col[key][0]):
			col[key] = (score, previous, boundary)

	def _start_context(self):
		return self.lm.start if self.lm else ()

	def _new_word(self, history, word):
		if self.lm:
			return (self.lm_weight * self.lm.logprob(word, history),
					self.lm.advance(history, word))
		return (-log(len(self.network.words)), ())

	def initialize(self, first_frame):
		self.columns = []
		self.stats = []
		col = {}
		cache = {}
		for state, entry_lp in self.network.starts:
			word = self.network.start_word[state]
			lm_lp, new_history = self._new_word(self._start_context(), word)
			key = (state, new_history)
			score = entry_lp + lm_lp + self.network.word_penalty
			score += self._score_emission(state, first_frame, cache)
			self._offer(col, key, score, None, True)
		if not col:
			raise RuntimeError('No starting states reachable')
		self.columns.append(col)
		self.stats.append({'frame':0, 'active':len(col),
				'transitions_considered':len(self.network.starts)})
		return col

	def add_column(self, frame):
		if not self.columns:
			return self.initialize(frame)
		net = self.network
		prev = self.columns[-1]
		candidates = {}
		count = 0
		for (state, history), (score, _, _) in prev.items():
			for edge in net.edges[state]:
				self._offer(candidates, (edge.dest, history),
						score + edge.logp, (state, history), False)
				count += 1

		# At word boundaries, extend each distinct LM history separately.
		# For a fixed history, only the best completed word path is needed.
		best_ends = {}
		for (state, history), (score, _, _) in prev.items():
			if state not in net.exit_logprob:
				continue
			candidate = score + net.exit_logprob[state]
			if history not in best_ends or candidate > best_ends[history][0]:
				best_ends[history] = (candidate, (state, history))
		for history, (score, predecessor) in best_ends.items():
			for state, entry_lp in net.starts:
				word = net.start_word[state]
				lm_lp, new_history = self._new_word(history, word)
				self._offer(candidates, (state, new_history),
					score + entry_lp + lm_lp + net.word_penalty,
					predecessor, True)
				count += 1

		col = {}
		cache = {}
		for key, (score, predecessor, boundary) in candidates.items():
			score += self._score_emission(key[0], frame, cache)
			self._offer(col, key, score, predecessor, boundary)
		if not col:
			raise RuntimeError(f'No states reachable at frame {len(self.columns)}')
		self.columns.append(col)
		self.stats.append({'frame':len(self.columns)-1, 'active':len(col),
				'transitions_considered':count})
		return col

	def build_viterbi(self, features):
		self.columns = []
		self.stats = []
		for frame in features:
			self.add_column(np.asarray(frame))
		return self

	def decode(self):
		if not self.columns:
			raise RuntimeError('Build the trellis first')
		last = self.columns[-1]
		best = (-np.inf, None)
		for key, (score, _, _) in last.items():
			state, history = key
			if state not in self.network.exit_logprob:
				continue
			end_lp = self.lm_weight * self.lm.logprob('</s>', history) if self.lm else 0.0
			score += self.network.exit_logprob[state] + end_lp
			if score > best[0]:
				best = (score, key)
		if best[1] is None:
			raise RuntimeError('No complete word at final frame')
		path = []
		words = []
		key = best[1]
		for t in range(len(self.columns)-1, -1, -1):
			_, predecessor, boundary = self.columns[t][key]
			path.append(key)
			if boundary:
				words.append(self.network.labels[key[0]][0])
			key = predecessor
		path.reverse()
		words.reverse()
		return {'words':words, 'transcription':' '.join(words),
			'score':float(best[0]), 'state_path':path}

	def show_columns(self, n=4, top=12):
		for t, col in enumerate(self.columns[:n]):
			print(f'Frame {t}: {len(col)} active (state, history) pairs')
			for key, (score, previous, boundary) in sorted(
					col.items(), key=lambda item:item[1][0], reverse=True)[:top]:
				state, history = key
				word, phone, occurrence, local = self.network.labels[state]
				print(f'  {word:>12s} {phone:>6s}[{occurrence},{local}]'
						f' history={history} score={score:.2f}'
						f' back={previous} word_start={boundary}')


def read_lexicon(path):
	lexicon = {}
	with open(path, encoding='utf-8-sig') as f:
		for line in f:
			if not line.strip():
				continue
			word, phones = line.rstrip('\r\n').split('\t', 1)
			lexicon[word] = phones.split()
	return lexicon
