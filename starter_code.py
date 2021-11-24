from src.extraction import start_processing_warcs
from src.entity_generation_ES import entity_generation
from src.context_vectors_2 import get_similarity_scores
import argparse
import pickle

def parse_cmd_arguments():
    cmd_parser =argparse.ArgumentParser(description='Parser for Entity Linking Program')

    cmd_parser.add_argument('-p', '--process_warcs', type=bool,
                        help="Required argument, write 'True' if you want to process the warc file(s), False otherwise")

    cmd_parser.add_argument('-s', '--save_es_results', type=bool,
                        help='Required argument, write 1 if you want to process and save the candidates '
                             'of the entity generation, 0 otherwise')

    parsed = cmd_parser.parse_known_args()[0]
    warc_bool = parsed.process_warcs
    es_bool = parsed.save_es_results

    option_parser = argparse.ArgumentParser()

    if warc_bool == 'False':
        option_parser.add_argument('-fp', '--filename_warcs', required=True,
                                   help="Required if -p == False, create a txt with the names of all the WARC picle files, "
                                        "seperated by a '\\n' that need to be imported in the program")
        filename_warcs = option_parser.parse_args()[0].filename_warcs
    else:
        filename_warcs = None

    return warc_bool, es_bool, filename_warcs

def read_all_warcs(list_of_warcs):
    list_of_texts = []
    for warc in list_of_warcs:
        with open(warc, "rb") as infile:
            texts = pickle.load(infile)
            list_of_texts.append(texts)

    return list_of_texts


def read_all_es_results(list_of_names):
    list_of_candidates = []
    for file in list_of_names:
        with open(file, "rb") as infile:
            candidates = pickle.load(infile)
            list_of_candidates.append(candidates)

    return list_of_candidates

def generate_and_save_entities(warcs):
    dict_of_candidates = {}

    for warc in warcs:
        for key, entities in warc.items():
            for entity_tuple in entities:
                mention, label, context = entity_tuple
                if mention not in dict_of_candidates.keys():
                    list_of_uris = entity_generation(mention, context)
                    dict_of_candidates[mention] = list_of_uris
                    print("Entity search completed for: ", mention)
                    print("Search Results:")
                    print(list_of_uris)
                    exit(1)

    with open('outputs/candidate_dictionary.pkl', 'wb') as f:
        pickle.dump(dict_of_candidates,f)

    return dict_of_candidates

if __name__ == '__main__':

    warc_bool, es_bool, fw = parse_cmd_arguments()

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


    #run_context_vector_script(warc_texts, list_of_candidates_per_warc)





    # with open(PKL_file, "rb") as infile:
    #     texts = pickle.load(infile)
    #
    # for key, entities in texts.items():
    #     for idx, entity_tuple in enumerate(entities):
    #         mention, label, context = entity_tuple
    #         list_of_uris = entity_generation("Washington",
    #                                          "George Washington (February 22, 1732 – December 14, 1799) was an American military officer, statesman, and Founding Father who served as the first president of the United States from 1789 to 1797")
    #         exit(1)

