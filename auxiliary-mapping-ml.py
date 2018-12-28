#This is used to map the auxiliary information (genre, director and actor) into mapping ID for MovieLens

import argparse

def mapping(fr_auxiliary, fw_mapping):
    '''
    mapping the auxiliary info (e.g., genre, director, actor) into ID

    Inputs:
        @fr_auxiliary: the auxiliary infomation
        @fw_mapping: the auxiliary mapping information
    '''
    actor_map = {}
    director_map = {}
    genre_map = {}

    actor_count = director_count = genre_count = 0

    for line in fr_auxiliary:

        lines = line.replace('\n', '').split('|')
        if len(lines) != 4: 
            continue
        
        movie_id = lines[0].split(':')[1]
        genre_list = []
        director_list = []
        actor_list = []

        for genre in lines[1].split(":")[1].split(','):
            if genre not in genre_map:
                genre_map.update({genre:genre_count})
                genre_list.append(genre_count)
                genre_count = genre_count + 1
            else:
                genre_id = genre_map[genre]
                genre_list.append(genre_id)
         
        for director in lines[2].split(":")[1].split(','):
            if director not in director_map:
                director_map.update({director:director_count})
                director_list.append(director_count)
                director_count = director_count + 1
            else:
                director_id = director_map[director]
                director_list.append(director_id)

        for actor in lines[3].split(':')[1].split(','):
            if actor not in actor_map:
                actor_map.update({actor:actor_count})
                actor_list.append(actor_count)
                actor_count = actor_count + 1
            else:
                actor_id = actor_map[actor]
                actor_list.append(actor_id)

        genre_list = ",".join(list(map(str, genre_list))) 
        director_list = ",".join(list(map(str, director_list))) 
        actor_list = ",".join(list(map(str, actor_list))) 

        output_line = movie_id + '|' + genre_list + '|' + director_list + '|' + actor_list + '\n'
        fw_mapping.write(output_line)

    return genre_count, director_count, actor_count


def print_statistic_info(genre_count, director_count, actor_count):
    '''
    print the number of genre, director and actor
    '''

    print ('The number of genre is: ' + str(genre_count))
    print ('The number of director is: ' + str(director_count))
    print ('The number of actor is: ' + str(actor_count))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=''' Map Auxiliary Information into ID''')

    parser.add_argument('--auxiliary', type=str, dest='auxiliary_file', default='data/ml/auxiliary.txt')
    parser.add_argument('--mapping', type=str, dest='mapping_file', default='data/ml/auxiliary-mapping.txt')

    parsed_args = parser.parse_args()

    auxiliary_file = parsed_args.auxiliary_file
    mapping_file = parsed_args.mapping_file
    
    fr_auxiliary = open(auxiliary_file,'r')
    fw_mapping = open(mapping_file,'w')

    genre_count, director_count, actor_count = mapping(fr_auxiliary, fw_mapping)
    print_statistic_info(genre_count, director_count, actor_count)

    fr_auxiliary.close()
    fw_mapping.close()