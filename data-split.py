#It aims to split the data into traning and testing according to the timestamp

import argparse
import operator



def round_int (rating_num, ratio):
    '''
    get the size of training data for each user

    Inputs:
        @rating_num: the total number of ratings for a specific user
        @ration: the percentage for training data

    Outputs:
        @train_size: the size of training data
    '''

    train_size = int(round(rating_num*ratio, 0))

    return train_size


def rank_rating_by_timestamp(fr_rating):
    '''
    rank the user-item rating data by the timestamp

    Inputs: 
        @fr_rating: the user-item rating data

    Outputs:
        @rank_data: user-specific rating data ranked by timestamp
    '''

    rank_data = {}

    for line in fr_rating:
        lines = line.split('\t')
        user = lines[0]
        item = lines[1]
        time = lines[3].replace('\n', '')
    
        item_list = []

        if user in rank_data:
            rank_data[user].update({item:time})
            
        else:
            rank_data.update({user:{item:time}})

    return rank_data

    
def split_rating_into_train_test(rank_data, fw_train, fw_test, ratio):
    '''
    split rank_rating data into training and test data

    Inputs:
        @rank_data: the ranked rating data
        @fw_train: the training data file
        @fw_test: the test data file
        @ratio: the percentage for training data
    '''

    for user in rank_data:
        item_list = rank_data[user]

        sorted_u = sorted(item_list.items(), key=operator.itemgetter(1))
        sorted_u = dict(sorted_u)
        
        rating_num = rank_data[user].__len__()
        train_size = round_int (rating_num, ratio)

        flag = 0

        for item in sorted_u:

            if flag < train_size:
                line = user+'\t'+ item + '\t' + sorted_u[item]+'\n'
                fw_train.write(line)
                flag = flag + 1
            else:
                line = user+'\t' + item + '\t' +sorted_u[item]+'\n'
                fw_test.write(line)
      
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=''' Split data into training and test''')

    parser.add_argument('--rating', type=str, dest='rating_file', default='data/rating-delete-missing-itemid.txt')
    parser.add_argument('--train', type=str, dest='train_file', default='data/training.txt')
    parser.add_argument('--test', type=str, dest='test_file', default='data/test.txt')
    parser.add_argument('--ratio', type=float, dest='ratio', default=0.8)

    parsed_args = parser.parse_args()

    rating_file = parsed_args.rating_file
    train_file = parsed_args.train_file
    test_file = parsed_args.test_file
    ratio = parsed_args.ratio
    
    fr_rating = open(rating_file,'r')
    fw_train = open(train_file,'w')
    fw_test = open(test_file,'w')

    rank_data = rank_rating_by_timestamp(fr_rating)
    split_rating_into_train_test(rank_data, fw_train, fw_test, ratio)

    fr_rating.close()
    fw_train.close()
    fw_test.close()