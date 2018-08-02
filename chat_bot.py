import numpy as np
from nltk.corpus import wordnet
from nltk import word_tokenize
import nltk
import retinasdk
from itertools import combinations
class chat_bot:
	def __init__(self, data_path):
		self.data = open(data_path, 'r').read()
		self.tokens = word_tokenize(self.data.lower())
		self.token_sets = list(set(self.tokens))
		self.pos_statement = nltk.pos_tag(nltk.tokenize.word_tokenize(self.data.lower()))
		self.pos_grammar = map(self.get_pos, self.pos_statement)
		self.data_size = len(self.data)
		self.vocab_size = len(list(set(self.pos_grammar)))
		

		self.hidden_size = 100 # size of hidden layer of neurons
		self.seq_length = 25 # number of steps to unroll the RNN for
		self.learning_rate = 1e-1
		# model parameters
		self.Wxh = np.random.randn(self.hidden_size, self.vocab_size)*0.01 # input to hidden
		self.Whh = np.random.randn(self.hidden_size, self.hidden_size)*0.01 # hidden to hidden
		self.Why = np.random.randn(self.vocab_size, self.hidden_size)*0.01 # hidden to output
		self.bh = np.zeros((self.hidden_size, 1)) # hidden bias
		self.by = np.zeros((self.vocab_size, 1)) # output bias

	def setDictionary(self, tokens, pos_grammar):
		this_dict={}  
		list_grammar = pos_grammar 
		
		for t in np.arange(len(tokens)):
			if(not this_dict.keys()):
				this_dict[list_grammar[t]]=[]
				this_dict[list_grammar[t]].append(tokens[t])

			if list_grammar[t] in this_dict.keys():
				this_dict[list_grammar[t]].append(tokens[t])
			else:
				this_dict[list_grammar[t]]=[]
				this_dict[list_grammar[t]].append(tokens[t])

		return this_dict

	def get_pos(self, pos_tag):
  		return(pos_tag[1])

	def lossFun(self, inputs, targets, hprev):
		xs, hs, ys, ps = {},{},{},{}
		hs[-1] = np.copy(hprev)
		loss = 0
		for t in range(len(inputs)):
			xs[t] = np.zeros((self.vocab_size, 1))
			xs[t][inputs[t]] = 1
			hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1])+self.bh)
			ys[t] = np.dot(self.Why, hs[t])+self.by
			ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
			
			loss += -np.log(ps[t][targets[t],0])
		dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
		dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
		dhnext = np.zeros_like(hs[0])
		for t in reversed(range(len(inputs))):
			dy = np.copy(ps[t])
			dy[targets[t]] -= 1
			dWhy += np.dot(dy, hs[t].T)
			dby += dy
			dh = np.dot(self.Why.T, dy) + dhnext
			dhraw = (1-hs[t]*hs[t])*dh
			dbh += dhraw
			dWxh += np.dot(dhraw, xs[t].T)
			dWhh += np.dot(dhraw, hs[t-1].T)
			dhnext = np.dot(self.Whh.T, dhraw)
		for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
			np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
		return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]
	def relu(self,x):
		return np.maximum(0.01*x, x)
	def drelu(self,x):
		x[x <= 0] = 0.01
		x[x > 0] = 1
		return x
	def sample(self, h, seed_ix):
		x = np.zeros((self.vocab_size, 1))
		x[seed_ix] = 1
		ixes = []
		i = 0
		while True:
			h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h)+ self.bh)
			y = np.dot(self.Why, h) + self.by
			p = np.exp(y) / np.sum(np.exp(y))
			ix = np.random.choice(range(self.vocab_size), p = p.ravel())				
			x = np.zeros((self.vocab_size, 1))

			if i == 0 and ix == self.check_pos('NNS'):
				x[ix] = 1
				ixes.append(ix)
				i += 1
				
			if i != 0:
				x[ix] = 1
				ixes.append(ix)
			if ix == self.check_pos('.') and i != 0:
				return ixes	
		
	def train(self):
		n, p = 0,0
		mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
		mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by)
		smooth_loss = -np.log(1.0/self.vocab_size)*self.seq_length
		loss1 = smooth_loss		
		self.pos_grammar = map(self.get_pos, self.pos_statement)
		grammar_to_ix = { gr:i for i, gr in enumerate(list(set(self.pos_grammar)))}		
		while True:
			
			self.pos_grammar = map(self.get_pos, self.pos_statement)
			if p+self.seq_length+1 >= len(list(self.pos_grammar)) or n ==0:
				hprev = np.zeros((self.hidden_size, 1))
				p = 0
			self.pos_grammar = map(self.get_pos, self.pos_statement)	
			inputs = [grammar_to_ix[gr] for gr in list(self.pos_grammar)[p:p+self.seq_length]]
			self.pos_grammar = map(self.get_pos, self.pos_statement)
			targets = [grammar_to_ix[gr] for gr in list(self.pos_grammar)[p+1:p+self.seq_length+1]]

			
			loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.lossFun(inputs, targets, hprev)
			smooth_loss = smooth_loss*0.999 +loss * 0.001
			if n % 100 == 0:

				print ('iter %d, loss: %f' % (n, smooth_loss))
			for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], 
                                		  [dWxh, dWhh, dWhy, dbh, dby], 
                                          [mWxh, mWhh, mWhy, mbh, mby]):
				mem += dparam * dparam
				param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)

			p += self.seq_length
			n += 1
			if ((loss1 - smooth_loss) > 60):
				#sample_ix = self.sample(hprev, inputs[0], 200)
				param = [self.Wxh, self.Whh, self.Why, self.bh, self.by]
				print(self.response_generator("hello, my name is DaeSung.", 10)) 
				return param

	def check_pos(self, word):
		self.pos_grammar = map(self.get_pos, self.pos_statement)
		grammar_to_ix = { gr:i for i, gr in enumerate(list(set(self.pos_grammar)))}	
		
		return grammar_to_ix[word]

	def check_gram(self, index):
		self.pos_grammar = map(self.get_pos, self.pos_statement)
		ix_to_grammar = { i:gr for i, gr in enumerate(list(set(self.pos_grammar)))}
		return ix_to_grammar[index]

	def keyword_classifier(self, sentence):
		liteClient = retinasdk.LiteClient("ba4c1950-95ec-11e8-917d-b5028d671452")
		return liteClient.getKeywords(sentence)

	def subsets(self, s):
		for cardinality in range(len(s)+1):
			yield from combinations(s, cardinality)

	def response_generator(self, sentence, num):
		keywords = self.keyword_classifier(sentence)
		sub_sets = [set(sub_set) for sub_set in self.subsets(keywords)]
		self.pos_grammar = map(self.get_pos, self.pos_statement)
		sentence_pos = 	nltk.pos_tag(nltk.tokenize.word_tokenize(sentence.lower()))	
		responses =[]
		hprev = np.zeros((self.hidden_size,1))
		while True:		
			for sub_set in sub_sets:
				temp = self.sample(hprev,self.check_pos("NN"))
				for keyword in sub_set:
					for index in range(len(temp)):
						if temp[index] == self.check_pos("NN") or temp[index] == self.check_pos("NNS"):
							temp[index] = keyword
							break
				responses.append(temp)				
				if len(responses) == num: break
			if len(responses) == num: break
		
				
		for i, response in enumerate(responses):
			for content in range(len(response)):
				if type(response[content]) is np.int32:
					self.pos_grammar = map(self.get_pos, self.pos_statement)		
					grammar_dict = self.setDictionary(self.tokens, list(self.pos_grammar))					
					dictionary = grammar_dict[self.check_gram(response[content])]
					responses[i][content] = dictionary[np.random.choice(len(dictionary))]					 
		return responses
	
				

chat_bot = chat_bot('C:\\Users\\User\\Desktop\\nlp-projects\\input3.txt')
hprev = np.zeros((100,1))
chat_bot.train()
#print(chat_bot.response_generator("hello, my name is DaeSung.", 10))
