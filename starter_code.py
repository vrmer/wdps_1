from entity_generation_ES import entity_generation
from src.context_vectors_2 import get_similarity_scores
import argparse
import pickle

PKL_FILE = 'outputs/CC-MAIN-20201001210429-20201002000429-00799_entities.pkl'

def parse_arguments():
    parser =argparse.ArgumentParser(description='Parser for Entity Linking Program')

    parser.add_argument('process_warcs', type=bool,
                        help='Required argument, write 1 if you want to process the warc file(s), 0 otherwise')

    # Optional positional argument
    parser.add_argument('opt_pos_arg', type=int, nargs='?',
                        help='An optional integer positional argument')

    # Optional argument
    parser.add_argument('--opt_arg', type=int,
                        help='An optional integer argument')

    # Switch
    parser.add_argument('--switch', action='store_true',
                        help='A boolean switch')


if __name__ == '__main__':

    parse_arguments()


    with open(PKL_file, "rb") as infile:
        texts = pickle.load(infile)

    for key, entities in texts.items():
        for idx, entity_tuple in enumerate(entities):
            mention, label, context = entity_tuple
            list_of_uris = entity_generation("Washington",
                                             "George Washington (February 22, 1732 – December 14, 1799) was an American military officer, statesman, and Founding Father who served as the first president of the United States from 1789 to 1797")
            exit(1)

