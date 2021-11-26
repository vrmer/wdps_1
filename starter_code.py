from collections import defaultdict
import fasttext
from src.extraction import start_processing_warcs
from src.entity_generation_ES import entity_generation
from src.candidate_selection import candidate_selection
import argparse
import pickle
import sys
import time

def str2bool(string):
    '''
    Converts the user input of any 'True' like string to boolean

    :param string: string to be converted
    :return: boolean
    '''
    return str(string).lower() in ("yes", "true", "t", "1")


def parse_cmd_arguments():
    '''
    Parses all command line arguments. See the help messages for more information

    :return: booleans and strings used later within the program
    '''

    cmd_parser =argparse.ArgumentParser(description='Parser for Entity Linking Program')

    cmd_parser.add_argument('-s', '--save_es_results', type=str,
                        help="Required argument, write 'True' if you want to process and save the candidates "
                             "of the entity generation, 'False' otherwise")

    cmd_parser.add_argument('-m', '--model_for_ranking', choices=['prominence','lesk','glove','bert'], required=False,
                            help="Required argument, write which model to use for candidate ranking. Possible options:"
                            "prominence | lesk | glove | bert, \n Default: prominence")

    cmd_parser.add_argument('-l', '--local', required=True,
                            help="Required argument, write 'True' if you want to run the local elastic search algorithm,"
                                 "'False' otherwise")

    cmd_parser.add_argument('-p', '--process_warcs', type=str, required= False,
                        help="Optional argument, write 'True' if you want to process the warc file(s), False otherwise")
    #
    # cmd_parser.add_argument('-a', '--archives_to_process', required=True,
    #                         help="Required argument, provide the filepath or filepaths to the WARC archives"
    #                              "you aim to process. Globbing is supported to allow you to go through multiple files.")

    cmd_parser.add_argument('-fp', '--filename_warcs', required='-p' in sys.argv,
                                   help="Required if -p == False, create a txt with the names of all the WARC pickle files, "
                                        "seperated by a '\\n' that need to be imported in the program")

    parsed = cmd_parser.parse_args()
    if parsed.process_warcs is None:
        warc_bool = True
    else:
        warc_bool = str2bool(parsed.process_warcs)

    es_bool = str2bool(parsed.save_es_results)
    local_bool = str2bool(parsed.local)

    if not warc_bool:
        filename_warcs = parsed.filename_warcs
    else:
        filename_warcs = None

    n_slices = parsed.n_slices

    return warc_bool, es_bool, filename_warcs, parsed.model_for_ranking, local_bool, n_slices


def read_all_warcs(list_of_warcs):
    '''
    Reads all warc files, given a list of warc file names from a txt file

    :param list_of_warcs: a list of warc file strings
    :return: warc dictionaries (key=webpage-id, value = list of tuples)
    '''

    list_of_texts = []
    for warc in list_of_warcs:
        with open("outputs/" + warc, "rb") as infile:
            texts = pickle.load(infile)
            list_of_texts.append(texts)

    return list_of_texts


def generate_and_save_entities(warcs, local_bool):
    '''
    Generates and saves the entities found through elasticsearch.
    Every 1000 entities, progress will be saved, so that stuff can still be tested if anything goes wrong.

    :param warcs: a list of all warc dictionaries (1 dict per warc file)
    :param local_bool: Indicates whether the elasticsearch should be run through a local or public client
    :return: a dict of all candidates
    '''

    print("Starting Entity Generation Process")

    dict_of_candidates = {}
    entities_checked = 0

    for warc in warcs:
        for key, entities in warc.items():
            for mention, label, context in entities:
                if mention not in dict_of_candidates.keys():
                    list_of_uris = entity_generation(mention, context, local_bool)
                    dict_of_candidates[mention] = list_of_uris
                    entities_checked +=1

                    if entities_checked % 200 ==0:
                        print("Entities checked: ", entities_checked)

                        if entities_checked % 1000 == 0:
                            print("Saving Temp Candidate_Dict")
                            with open('outputs/temp_candidate_dictionary.pkl', 'wb') as f:
                                pickle.dump(dict_of_candidates,f)

    with open('outputs/candidate_dictionary.pkl', 'wb') as f:
        pickle.dump(dict_of_candidates, f)

    return dict_of_candidates

def disambiguate_entities( warc_texts, candidate_dict, method='popularity' ):
    '''
    Tries to rank the URI's by disambiguating the entities obtained through ElasticSearch

    :param warc_texts: a list of dictionaries containing the webpage-id's and corresponding entities
    :param candidate_dict: a dictionary with all results of the ElasticSearch
    :param method: The method to be used, the default is 'popularity'
    :return: returns a list of triples, containing the web page id, the entity and the wikidata URI
    '''

    start = time.perf_counter()

    output = candidate_selection(warc_texts,candidate_dict,method)

    end = time.perf_counter()

    total_time = end - start
    print(f'Total time spent in disambiguating: {total_time}')

    return output

if __name__ == '__main__':
    '''
    Main program
    Structure of the main:
        - Parses commandline/terminal input
        - Performs a parallel elastic search
        - Write results in corresponding files
    '''

    lang_det = fasttext.load_model('lid.176.ftz')

    warc_bool, es_bool, fw, model, local_bool, n_slices = parse_cmd_arguments()

    if warc_bool:
        list_of_warcnames = start_processing_warcs(warc_bool) ### TODO: to change
    else:
        with open(fw) as f:
            list_of_warcnames = list(f.readlines())

    warc_texts = read_all_warcs(list_of_warcnames)

    if es_bool:
        candidate_dict = generate_and_save_entities(warc_texts, local_bool)
    else:
        with open("candidate_dictionary.pkl", "rb") as f:
            candidate_dict = pickle.load(f)

    for idx, warc in enumerate(warc_texts):
        output = disambiguate_entities(warc, candidate_dict, model)

        with open(f"results/annotations_{list_of_warcnames[idx][:-4]}_{model}", 'w') as outfile:
            for entity_tuple in output:
                outfile.write(entity_tuple[0] + '\t' + entity_tuple[1] + '\t' + entity_tuple[2] + '\n')
