import numpy as np
import argparse
import torch
import torch.autograd as autograd
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from LSTMTagger import LSTMTagger
from LSTMTrain import LSTMTrain


def load_paths(fr_file, isPositive):
	'''
	load postive or negative paths, map all nodes in paths into ids

	Inputs:
		@fr_file: positive or negative paths
		@isPositive: identify fr_file is positive or negative
	'''

	global node_count, all_variables, paths_between_pairs, positive_label, all_user, all_movie
	
	for line in fr_file:
		line = line.replace('\n', '')
		lines = line.split(',')
		user = lines[0]
		movie = lines[-1]

		if user not in all_user:
			all_user.append(user)
		if movie not in all_movie:
			all_movie.append(movie)

		key = (user, movie)
		value = []
		path = []

		if isPositive:
			if key not in positive_label:
				positive_label.append(key)

		for node in lines:
			if node not in all_variables:
				all_variables.update({node:node_count})
				node_count = node_count + 1
			path.append(node)

		if key not in paths_between_pairs:
			value.append(path)
			paths_between_pairs.update({key:value})
		else:
			paths_between_pairs[key].append(path)


def load_pre_embedding(fr_pre_file, isUser):
	'''
	load pre-train-user or movie embeddings

	Inputs:
		@fr_pre_file: pre-train-user or -movie embeddings
		@isUser: identify user or movie
	'''
	global pre_embedding, all_variables

	for line in fr_pre_file:
		lines = line.split('|')
		node = lines[0]
		if isUser:
			node = 'u' + node
		else:
			node = 'i' + node

		if node in all_variables:
			node_id = all_variables[node]
			embedding = [float(x) for x in lines[1].split()]
			embedding = np.array(embedding)
			pre_embedding[node_id] = embedding


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description=''' Recurrent Neural Network ''')

	parser.add_argument('--inputdim', type=int, dest='input_dim', default=10)
	parser.add_argument('--hiddendim', type=int, dest='hidden_dim', default=16)
	parser.add_argument('--outdim', type=int, dest='out_dim', default=1)
	parser.add_argument('--iteration', type=int, dest='iteration', default=5)
	parser.add_argument('--learingrate', type=float, dest='learning_rate', default=0.001)
	parser.add_argument('--positivepath', type=str, dest='positive_path', default='data/ml/positive-path.txt')
	parser.add_argument('--negativepath', type=str, dest='negative_path', default='data/ml/negative-path.txt')
	parser.add_argument('--pretrainuserembedding', type=str, dest='pre_train_user_embedding', default='data/ml/pre-train-user-embedding.txt')
	parser.add_argument('--pretrainmovieembedding', type=str, dest='pre_train_movie_embedding', default='data/ml/pre-train-movie-embedding.txt')
	parser.add_argument('--posttrainembedding', type=str, dest='post_train_embedding', default='data/ml/post-train-embedding.txt')

	parsed_args = parser.parse_args()

	input_dim = parsed_args.input_dim
	hidden_dim = parsed_args.hidden_dim
	out_dim = parsed_args.out_dim
	iteration = parsed_args.iteration
	learning_rate = parsed_args.learning_rate

	positive_path = parsed_args.positive_path
	negative_path = parsed_args.negative_path
	pre_train_user_embedding = parsed_args.pre_train_user_embedding
	pre_train_movie_embedding = parsed_args.pre_train_movie_embedding
	post_train_embedding = parsed_args.post_train_embedding


	fr_postive = open(positive_path, 'r')
	fr_negative = open(negative_path, 'r')
	fr_pre_user = open(pre_train_user_embedding, 'r')
	fr_pre_movie = open(pre_train_movie_embedding, 'r')
	fw_post_train = open(post_train_embedding, 'w')

	node_count = 0 #count the number of all entities (user, movie and attributes)
	all_variables = {} #save variable and corresponding id
	paths_between_pairs = {} #save all the paths (both positive and negative) between a user-movie pair
	positive_label = [] #save the positive user-movie pairs
	all_user = [] #save all the users
	all_movie = [] #save all the movies


	load_paths(fr_postive, True)
	load_paths(fr_negative, False)
	print ('The number of all variables is :' + str(len(all_variables)))

	node_size = len(all_variables)
	pre_embedding = np.random.rand(node_size, input_dim) #embeddings for all nodes
	load_pre_embedding(fr_pre_user, True)
	load_pre_embedding(fr_pre_movie, False)
	pre_embedding = torch.FloatTensor(pre_embedding)

	model = LSTMTagger(node_size, input_dim, hidden_dim, out_dim, pre_embedding)
	if torch.cuda.is_available():
	    model = model.cuda()

	model_train = LSTMTrain(model, iteration, learning_rate, paths_between_pairs, positive_label, \
	    all_variables, all_user, all_movie, fw_post_train)
	model_train.train()

	fr_postive.close()
	fr_negative.close()
	fr_pre_user.close()
	fr_pre_movie.close()
	fw_post_train.close()