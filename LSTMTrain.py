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
		all_variables, all_user, all_movie, fw_post_train):
		super(LSTMTrain, self).__init__()
		self.model = model
		self.iteration = iteration
		self.learning_rate = learning_rate
		self.paths_between_pairs = paths_between_pairs
		self.positive_label = positive_label
		self.all_variables = all_variables
		self.all_user = all_user
		self.all_movie = all_movie
		self.fw_post_train = fw_post_train


	def dump_post_embedding(self, fw_post_train, isUser):
		'''
		dump the post-train user or item embedding

		Inputs:
			@fw_post_file: post-train-user and -item embeddings
			@isUser: identify user or movie
		'''
		node_list = self.all_user + self.all_item

		for node in node_list:
			node_id = torch.LongTensor([int(self.all_variables[node])])
			node_id = Variable(node_id)
			if torch.cuda.is_available():
				ur_id = ur_id.cuda()
			node_embedding = self.model.embedding(node_id).squeeze().cpu().data.numpy()

			node_embedding_str = [str(x) for x in node_embedding]
			node_embedding_str = " ".join(node_embedding_str)
			output = node + '|' + node_embedding_str + '\n'
			fw_post_train.write(output)

		
	def train(self):
		criterion = nn.BCELoss()
		optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

		for epoch in range (self.iteration):
			running_loss = 0.0
			running_acc = 0.0
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
				
				pred = torch.round(out)
				num_correct = (pred.cpu() == label).sum()
				running_acc += num_correct.item()
				
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			
			print('epoch['+str(epoch) + ']: loss is '+str(running_loss))
			print('accuracy is: '+ str(running_acc/data_size))

		self.dump_post_embedding(self.fw_post_train)