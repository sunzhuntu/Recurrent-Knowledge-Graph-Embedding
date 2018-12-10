#Build knowledge graph and mine the connected paths between users and items in the training data

import argparse
import networkx as nx
import random


def load_data(file):
    '''
    load training (positive) or negative user-movie interaction data

    Input:
        @file: training (positive) data or negative data

    Output:
        @data: pairs containing positive or negative interaction data  
    '''
    data = []

    for line in file:
        lines = line.split('\t')
        user = lines[0]
        movie = lines[1].replace('\n','')
        data.append((user, movie))

    return data


def add_user_movie_interaction_into_graph(positive_rating):
    '''
    add user-movie interaction data into the graph

    Input:
        @pos_rating: user-movie interaction data

    Output:
        @Graph: the built graph with user-movie interaction info 
    '''  
    Graph = nx.DiGraph()       

    for pair in positive_rating:
        user = pair[0]
        movie = pair[1]
        user_node = 'u' + user
        movie_node = 'i' + movie
        Graph.add_node(user_node)
        Graph.add_node(movie_node)
        Graph.add_edge(user_node, movie_node)

    return Graph


def add_auxiliary_into_graph(fr_auxiliary, Graph):
    '''
    add auxiliary information (e.g., actor, director, genre) into graph

    Input:
        @fr_auxiliary: auxiliary mapping information about movies
        @Graph: the graph with user-movie interaction info

    Output:
        @Graph: the graph with user-moive interaction and auxiliary info
    '''

    for line in fr_auxiliary:
        lines = line.split('|')
        if len(lines) != 4: continue
        
        movie_id = lines[0]
        genre_list = lines[1].split(',')
        director_list = lines[2].split(',')
        actor_list = lines[3].split(',')

        #add movie nodes into Graph, in case the movie is not included in the training data
        movie_node = 'i' + movie_id
        if not Graph.has_node(movie_node):
            Graph.add_node(movie_node)

        #add the genre nodes into the graph;  
        #as genre connection is too dense, we add one genre to avoid over-emphasizing its effect
        genre_id = genre_list[0]
        genre_node = 'g' + genre_id
        if not Graph.has_node(genre_node):
            Graph.add_node(genre_node)
        Graph.add_edge(movie_node, genre_node)
        Graph.add_edge(genre_node, movie_node)

        #add the director nodes into the graph
        for director_id in director_list:
            director_node = 'd' + director_id
            if not Graph.has_node(director_node):
                Graph.add_node(director_node)
            Graph.add_edge(movie_node, director_node)
            Graph.add_edge(director_node, movie_node)

        #add the actor nodes into the graph
        for actor_id in actor_list:
            actor_node = 'a' + actor_id
            if not Graph.has_node(actor_node):
                Graph.add_node(actor_node)
            Graph.add_edge(movie_node, actor_node)
            Graph.add_edge(actor_node, movie_node)

    return Graph


def print_graph_statistic(Graph):
    '''
    output the statistic info of the graph

    Input:
        @Graph: the built graph 
    '''
    print('The knowledge graph has been built completely \n')
    print('The number of nodes is:  ' + str(len(Graph.nodes()))+ ' \n') 
    print('The number of edges is  ' + str(len(Graph.edges()))+ ' \n')


def mine_paths_between_nodes(Graph, user_node, movie_node, maxLen, sample_size, fw_file):
    '''
    mine qualified paths between user and movie nodes, and get sampled paths between nodes

    Inputs:
        @user_node: user node
        @movie_node: movie node
        @maxLen: path length
        @fw_file: the output file for the mined paths
    ''' 

    connected_path = [] 
    for path in nx.all_simple_paths(Graph, source=user_node, target=movie_node, cutoff=maxLen):
        if len(path) == maxLen + 1: 
            connected_path.append(path)

    path_size = len(connected_path)

    #as there is a huge number of paths connected user-movie nodes, we get randomly sampled paths
    #random sample can better balance the data distribution and model complexity
    if path_size > sample_size:
        random.shuffle(connected_path)
        connected_path = connected_path[:sample_size]
    
    for path in connected_path:
        fw_file.write(str(path) + '\n')
   
    #print('The number of paths between '+ user_node + ' and ' + movie_node + ' is: ' +  str(len(connected_path)) +'\n')


def dump_paths(Graph, rating_pair, maxLen, sample_size, fw_file):
    '''
    dump the postive or negative paths 

    Inputs:
        @Graph: the well-built knowledge graph
        @rating_pair: positive_rating or negative_rating
        @maxLen: path length
        @sample_size: size of sampled paths between user-movie nodes
    '''
    for pair in rating_pair:
        user_id = pair[0]
        movie_id = pair[1]
        user_node = 'u' + user_id
        movie_node = 'i' + movie_id

        if Graph.has_node(user_node) and Graph.has_node(movie_node):
            mine_paths_between_nodes(Graph, user_node, movie_node, maxLen, sample_size, fw_file)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=''' Build Knowledge Graph and Mine the Connected Paths''')

    parser.add_argument('--training', type=str, dest='training_file', default='data/ml/training.txt')
    parser.add_argument('--negtive', type=str, dest='negative_file', default='data/ml/negative.txt')
    parser.add_argument('--auxiliary', type=str, dest='auxiliary_file', default='data/ml/auxiliary-mapping.txt')
    parser.add_argument('--positivepath', type=str, dest='positive_path', default='data/ml/positive-path.txt', \
                        help='paths between user-item interaction pairs')
    parser.add_argument('--negativepath', type=str, dest='negative_path', default='data/ml/negative-path.txt', \
                        help='paths between negative sampled user-item pair')
    parser.add_argument('--pathlength', type=int, dest='path_length', default=3, help='length of paths with choices [3,5,7]')
    parser.add_argument('--samplesize', type=int, dest='sample_size', default=5, help='the sampled size of paths bwteen nodes')

    parsed_args = parser.parse_args()

    training_file = parsed_args.training_file
    negative_file = parsed_args.negative_file
    auxiliary_file = parsed_args.auxiliary_file
    positive_path = parsed_args.positive_path
    negative_path = parsed_args.negative_path
    path_length = parsed_args.path_length
    sample_size = parsed_args.sample_size


    fr_training = open(training_file,'r')
    fr_negative = open(negative_file, 'r')
    fr_auxiliary = open(auxiliary_file,'r')
    fw_positive_path = open(positive_path, 'w')
    fw_negative_path = open(negative_path, 'w')

    positive_rating = load_data(fr_training)
    negative_rating = load_data(fr_negative)

    print('The number of user-movie interaction data is:  ' + str(len(positive_rating))+ ' \n')
    print('The number of negative sampled data is:  ' + str(len(negative_rating))+ ' \n') 

    Graph = add_user_movie_interaction_into_graph(positive_rating)
    Graph = add_auxiliary_into_graph(fr_auxiliary, Graph)
    print_graph_statistic(Graph)

    dump_paths(Graph, positive_rating, path_length, sample_size, fw_positive_path)
    dump_paths(Graph, negative_rating, path_length, sample_size, fw_negative_path)


    fr_training.close()
    fr_negative.close()
    fr_auxiliary.close()
    fw_positive_path.close()
    fw_negative_path.close()