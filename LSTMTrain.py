import numpy as np
import torch
import torch.autograd as autograd
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

class LSTMTrain(object):
	'''
	recurrent neural network training process
	'''
	def __init__(self, model, iteration, learning_rate, paths_between_pairs, positive_label, \
		all_variables, all_user, all_movie):
		super(LSTMTrain, self).__init__()
		self.model = model
		self.iteration = iteration
		self.learning_rate = learning_rate
		self.paths_between_pairs = paths_between_pairs
		self.positive_label = positive_label
		self.all_variables = all_variables
		self.all_user = all_user
		self.all_movie = all_movie


	def dump_post_embedding(self):
		'''
		dump the post-train user or item embedding
		'''
		embedding_dict = {}
		node_list = self.all_user + self.all_movie

		for node in node_list:
			node_id = torch.LongTensor([int(self.all_variables[node])])
			node_id = Variable(node_id)
			if torch.cuda.is_available():
				ur_id = ur_id.cuda()
			node_embedding = self.model.embedding(node_id).squeeze().cpu().data.numpy()
			if node not in embedding_dict:
				embedding_dict.update({node:node_embedding})

		return embedding_dict
			
		
	def train(self):
		criterion = nn.BCELoss()
		#You may also try different types of optimization methods (e.g, SGD, RMSprop, Adam, Adadelta, etc.)
		optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
		#optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)

		for epoch in range (self.iteration):
			running_loss = 0.0
			data_size = len(self.paths_between_pairs)
			label = Variable(torch.Tensor())    
			
			for pair in self.paths_between_pairs:
				
				paths_between_one_pair = self.paths_between_pairs[pair]
				paths_between_one_pair_size = len(paths_between_one_pair)
				paths_between_one_pair_id = []

				for path in paths_between_one_pair:
					path_id = [self.all_variables[x] for x in path]
					paths_between_one_pair_id.append(path_id) 
			
				paths_between_one_pair_id = np.array(paths_between_one_pair_id)
				paths_between_one_pair_id = Variable(torch.LongTensor(paths_between_one_pair_id))
					
				
				if torch.cuda.is_available():
					paths_between_one_pair_id = paths_between_one_pair_id.cuda()
				
				out = self.model(paths_between_one_pair_id)
				out = out.squeeze()
			
				if pair in self.positive_label:
					label = Variable(torch.Tensor([1]))
				else:
					label = Variable(torch.Tensor([0])) 
			
				loss = criterion(out.cpu(), label)
				running_loss += loss.item() * label.item()
				
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			
			print('epoch['+str(epoch) + ']: loss is '+str(running_loss))

		return self.dump_post_embedding()