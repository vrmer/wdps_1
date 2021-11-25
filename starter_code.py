from src.extraction import start_processing_warcs
from src.entity_generation_ES import entity_generation
from src.candidate_selection import candidate_selection
import argparse
import pickle
import sys
import time

def str2bool(v):
  return str(v).lower() in ("yes", "true", "t", "1")

def parse_cmd_arguments():

    cmd_parser =argparse.ArgumentParser(description='Parser for Entity Linking Program')

    cmd_parser.add_argument('-s', '--save_es_results', type=str,
                        help='Required argument, write 1 if you want to process and save the candidates '
                             'of the entity generation, 0 otherwise')

    cmd_parser.add_argument('-p', '--process_warcs', type=str, required= False,
                        help="Optional argument, write 'True' if you want to process the warc file(s), False otherwise")

    cmd_parser.add_argument('-fp', '--filename_warcs', required='-p' in sys.argv,
                                   help="Required if -p == False, create a txt with the names of all the WARC picle files, "
                                        "seperated by a '\\n' that need to be imported in the program")

    cmd_parser.add_argument('-model', '--model_for_ranking', choices=['popularity','lesk','glove','bert'], required=True,
                            help="Required argument, write which model to use for candidate ranking. Possible options:"
                            "popularity | lesk | glove | bert")


    parsed = cmd_parser.parse_args()
    if parsed.process_warcs is None:
        warc_bool = True
    else:
        warc_bool = str2bool(parsed.process_warcs)

    es_bool = str2bool(parsed.save_es_results)

    if not warc_bool:
        filename_warcs = parsed.filename_warcs
    else:
        filename_warcs = None

    return warc_bool, es_bool, filename_warcs, parsed.model_for_ranking


def read_all_warcs(list_of_warcs):
    list_of_texts = []
    for warc in list_of_warcs:
        with open("outputs/" + warc, "rb") as infile:
            texts = pickle.load(infile)
            list_of_texts.append(texts)

    return list_of_texts


def read_all_es_results(list_of_names):
    list_of_candidates = []
    for file in list_of_names:
        with open("outputs/" + file, "rb") as infile:
            candidates = pickle.load(infile)
            list_of_candidates.append(candidates)

    return list_of_candidates

def generate_and_save_entities(warcs):
    dict_of_candidates = {}

    start = time.time()
    idx = 0

    for warc in warcs:
        for key, entities in warc.items():
            for mention, label, context in entities:
                if mention not in dict_of_candidates.keys():
                    list_of_uris = entity_generation(mention, context)
                    dict_of_candidates[mention] = list_of_uris
                    print("Entity search completed for: ", mention)
                    print("Best Result:", list_of_uris if not list_of_uris else list_of_uris[0])

                    if idx > 200:
                        print(time.time()-start)
                        exit(1)
                    else:
                        idx +=1


    with open('outputs/candidate_dictionary.pkl', 'wb') as f:
        pickle.dump(dict_of_candidates,f)

    return dict_of_candidates

def disambiguate_entities (warc_texts, candidate_dict,method='popularity'):

    start = time.perf_counter()

    output = candidate_selection(warc_texts,candidate_dict,method)

    end = time.perf_counter()

    total_time = end - start
    print(f'Total time spent in disambiguating: {total_time}')

    return output

if __name__ == '__main__':

    warc_bool, es_bool, fw, model_for_selection = parse_cmd_arguments()

    if warc_bool:
        list_of_warcnames = start_processing_warcs()
    else:
        with open(fw) as f:
            list_of_warcnames = list(f.readlines())

    warc_texts = read_all_warcs(list_of_warcnames)

    if es_bool:
        candidate_dict = generate_and_save_entities(warc_texts)
    else:
        with open("outputs/candidate_dictionary.pkl", "rb") as f:
            candidate_dict = pickle.load(f)

    output = disambiguate_entities(warc_texts, candidate_dict, model_for_selection)
    with open("outputs/output.txt", 'w') as outfile:
        for entity_tuple in output:
            outfile.write(f'{entity_tuple[0]} {entity_tuple[1]}{chr(10)}')  # chr 10 = new line

    # testing score.py

    # with open(PKL_file, "rb") as infile:
    #     texts = pickle.load(infile)
    #
    # for key, entities in texts.items():
    #     for idx, entity_tuple in enumerate(entities):
    #         mention, label, context = entity_tuple
    #         list_of_uris = entity_generation("Washington",
    #                                          "George Washington (February 22, 1732 – December 14, 1799) was an American military officer, statesman, and Founding Father who served as the first president of the United States from 1789 to 1797")
    #         exit(1)

