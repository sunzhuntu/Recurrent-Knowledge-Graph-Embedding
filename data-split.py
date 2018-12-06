#This is used to split the user-movie interaction data into traning and test according to the timestamp

import argparse
import operator

def round_int (rating_num, ratio):
    '''
    get the size of training data for each user

    Inputs:
        @rating_num: the total number of ratings for a specific user
        @ration: the percentage for training data

    Output:
        @train_size: the size of training data
    '''

    train_size = int(round(rating_num*ratio, 0))

    return train_size


def load_data(fr_rating):
    '''
    load the user-item rating data with the timestamp

    Input: 
        @fr_rating: the user-item rating data

    Output:
        @rating_data: user-specific rating data with timestamp
    '''

    rating_data = {}

    for line in fr_rating:
        lines = line.split('\t')
        user = lines[0]
        item = lines[1]
        time = lines[3].replace('\n', '')
    
        item_list = []

        if user in rating_data:
            rating_data[user].update({item:time})
            
        else:
            rating_data.update({user:{item:time}})

    return rating_data

    
def split_rating_into_train_test(rating_data, fw_train, fw_test, ratio):
    '''
    split rating_rating data into training and test data by timestamp

    Inputs:
        @rating_data: the user-specific rating data
        @fw_train: the training data file
        @fw_test: the test data file
        @ratio: the percentage of training data
    '''

    for user in rating_data:
        item_list = rating_data[user]

        sorted_u = sorted(item_list.items(), key=operator.itemgetter(1))
        sorted_u = dict(sorted_u)
        
        rating_num = rating_data[user].__len__()
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

    parser.add_argument('--rating', type=str, dest='rating_file', default='data/ml/rating-delete-missing-itemid.txt')
    parser.add_argument('--train', type=str, dest='train_file', default='data/ml/training.txt')
    parser.add_argument('--test', type=str, dest='test_file', default='data/ml/test.txt')
    parser.add_argument('--ratio', type=float, dest='ratio', default=0.8)

    parsed_args = parser.parse_args()

    rating_file = parsed_args.rating_file
    train_file = parsed_args.train_file
    test_file = parsed_args.test_file
    ratio = parsed_args.ratio
    
    fr_rating = open(rating_file,'r')
    fw_train = open(train_file,'w')
    fw_test = open(test_file,'w')

    rating_data = load_data(fr_rating)
    split_rating_into_train_test(rating_data, fw_train, fw_test, ratio)

    fr_rating.close()
    fw_train.close()
    fw_test.close()