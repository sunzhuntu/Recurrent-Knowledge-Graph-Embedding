#This is part is aims to feed connected paths into the recurrent neural network to train and test the proposed methods.

import numpy as np
import argparse
import torch
import torch.autograd as autograd
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from LSTMTagger import LSTMTagger
from LSTMTrain import LSTMTrain
from LSTMEvaluation import LSTMEvaluation
from datetime import datetime


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


def load_data(fr_file):
	'''
	load training or test data

	Input: 
			@fr_rating: the user-item rating data

	Output:
			@rating_data: user-specific rating data with timestamp
	'''

	data_dict = {}

	for line in fr_file:
			lines = line.replace('\n', '').split('\t')
			user = 'u' + lines[0]
			item = 'i' + lines[1]

			if user not in data_dict:
				data_dict.update({user:[item]})
			elif item not in data_dict[user]:
				data_dict[user].append(item)

	return data_dict


def write_results(fw_results, precision_1, precision_5, precision_10, mrr_10):
	'''
	write results into text file
	'''
	line = 'precision@1: ' + str(precision_1) + '\n' + 'precision@5: ' + str(precision_5) + '\n' \
		+ 'precision@10: ' + str(precision_10) + '\n' + 'mrr: ' + str(mrr_10) + '\n'
	fw_results.write(line)



if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description=''' Recurrent Neural Network ''')

	'''
	Parameter Settings: 
	for MovieLens in terms of [input_dim, hidden_dim, out_dim, iteration, learning_rate, optimizer] is [10, 16, 1, 5, 0.2/0.1, SGD]
	for Yelp in terms of [input_dim, hidden_dim, out_dim, iteration, learning_rate, optimizer] is [20, 32, 1, 5, 0.01, SGD]
	You can change optimizer in the LSTMTrain class
	'''

	parser.add_argument('--inputdim', type=int, dest='input_dim', default=10)
	parser.add_argument('--hiddendim', type=int, dest='hidden_dim', default=16)
	parser.add_argument('--outdim', type=int, dest='out_dim', default=1)
	parser.add_argument('--iteration', type=int, dest='iteration', default=5)
	parser.add_argument('--learingrate', type=float, dest='learning_rate', default=0.2)
	
	parser.add_argument('--positivepath', type=str, dest='positive_path', default='data/ml/positive-path.txt')
	parser.add_argument('--negativepath', type=str, dest='negative_path', default='data/ml/negative-path.txt')
	parser.add_argument('--pretrainuserembedding', type=str, dest='pre_train_user_embedding', default='data/ml/pre-train-user-embedding.txt')
	parser.add_argument('--pretrainmovieembedding', type=str, dest='pre_train_movie_embedding', default='data/ml/pre-train-item-embedding.txt')
	parser.add_argument('--train', type=str, dest='train_file', default='data/ml/training.txt')
	parser.add_argument('--test', type=str, dest='test_file', default='data/ml/test.txt')
	parser.add_argument('--results', type=str, dest='results', default='data/ml/results.txt')

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
	train_file = parsed_args.train_file
	test_file = parsed_args.test_file
	results_file = parsed_args.results

	start_time = datetime.now()

	fr_postive = open(positive_path, 'r')
	fr_negative = open(negative_path, 'r')
	fr_pre_user = open(pre_train_user_embedding, 'r')
	fr_pre_movie = open(pre_train_movie_embedding, 'r')
	fr_train = open(train_file,'r')
	fr_test = open(test_file,'r')
	fw_results = open(results_file, 'w')


	node_count = 0 #count the number of all entities (user, movie and attributes)
	all_variables = {} #save variable and corresponding id
	paths_between_pairs = {} #save all the paths (both positive and negative) between a user-movie pair
	positive_label = [] #save the positive user-movie pairs
	all_user = [] #save all the users
	all_movie = [] #save all the movies

	start_time = datetime.now()
	load_paths(fr_postive, True)
	load_paths(fr_negative, False)
	print ('The number of all variables is :' + str(len(all_variables)))
	end_time = datetime.now()
	duration = end_time - start_time
	print ('the duration for loading user path is ' + str(duration) + '\n')

	start_time = datetime.now()
	node_size = len(all_variables)
	pre_embedding = np.random.rand(node_size, input_dim) #embeddings for all nodes
	load_pre_embedding(fr_pre_user, True)
	load_pre_embedding(fr_pre_movie, False)
	pre_embedding = torch.FloatTensor(pre_embedding)
	end_time = datetime.now()
	duration = end_time - start_time
	print ('the duration for loading embedding is ' + str(duration) + '\n')

	start_time = datetime.now()
	model = LSTMTagger(node_size, input_dim, hidden_dim, out_dim, pre_embedding)
	if torch.cuda.is_available():
		model = model.cuda()

	model_train = LSTMTrain(model, iteration, learning_rate, paths_between_pairs, positive_label, \
		all_variables, all_user, all_movie)
	embedding_dict = model_train.train()
	print('model training finished')
	end_time = datetime.now()
	duration = end_time - start_time
	print ('the duration for model training is ' + str(duration) + '\n')

	start_time = datetime.now()
	train_dict = load_data(fr_train)
	test_dict = load_data(fr_test)
	model_evaluation = LSTMEvaluation(embedding_dict, all_movie, train_dict, test_dict)
	top_score_dict = model_evaluation.calculate_ranking_score()
	precision_1,_ = model_evaluation.calculate_results(top_score_dict, 1)
	precision_5,_ = model_evaluation.calculate_results(top_score_dict, 5)
	precision_10, mrr_10 = model_evaluation.calculate_results(top_score_dict, 10)
	end_time = datetime.now()
	duration = end_time - start_time
	print ('the duration for model evaluation is ' + str(duration) + '\n')

	write_results(fw_results, precision_1, precision_5, precision_10, mrr_10)

	end_time = datetime.now()
	duration = end_time - start_time
	print ('the duration for loading item embedding is ' + str(duration) + '\n')

	fr_postive.close()
	fr_negative.close()
	fr_pre_user.close()
	fr_pre_movie.close()
	fr_train.close()
	fr_test.close()
	fw_results.close()