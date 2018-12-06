#This is used to split the user-movie interaction data into traning and test according to the timestamp

import argparse
import math
from random import randint

def load_data (file):
    '''
    load training data

    Input:
        @file: training data

    Outputs:
        @train_dict: user-specific training data
        @item_list: all items in the training data
    '''
    train_dict = {}
    all_movie_list = []

    for line in file:
        lines = line.split('\t')
        user = lines[0]
        movie = lines[1].replace('\n','')
        
        if user not in train_dict:
            init_movie_list = []
            init_movie_list.append(movie)
            train_dict.update({user:init_movie_list})
        else:
            train_dict[user].append(movie)
        
        if movie not in all_movie_list:
            all_movie_list.append(movie)

    return train_dict, all_movie_list


def negative_sample(train_dict, all_movie_list, shrink, fw_negative):
    '''
    sample negative movies for all users in training data

    Inputs:
        @train_dict: user-specific training data
        @all_movie_list: all the movies in the training data
    '''
    all_movie_size = len(all_movie_list)

    for user in train_dict:
        user_train_movie = train_dict[user]
        user_train_movie_size = len(user_train_movie)
        negative_size = math.ceil(user_train_movie_size * shrink)
        user_negative_movie = []

        while (len(user_negative_movie) < negative_size):
            negative_index = randint(0, (all_movie_size - 1))
            negative_movie = str(all_movie_list[negative_index])
            if negative_movie not in user_train_movie and negative_movie not in user_negative_movie:
                user_negative_movie.append(negative_movie)
                line = user + '\t' + negative_movie + '\n'
                fw_negative.write(line)
      
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=''' Sample Negative Movies for Each User''')

    parser.add_argument('--train', type=str, dest='train_file', default='data/ml/training.txt')
    parser.add_argument('--negative', type=str, dest='negative_file', default='data/ml/negative.txt')
    parser.add_argument('--shrink', type=float, dest='shrink', default=0.05)

    parsed_args = parser.parse_args()

    train_file = parsed_args.train_file
    negative_file = parsed_args.negative_file
    shrink = parsed_args.shrink
    
    fr_train = open(train_file,'r')
    fw_negative = open(negative_file,'w')

    train_dict, all_movie_list = load_data(fr_train)
    negative_sample(train_dict, all_movie_list, shrink, fw_negative)


    fr_train.close()
    fw_negative.close()