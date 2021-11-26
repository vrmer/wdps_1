import fasttext
from src.extraction import start_processing_warcs
from src.entity_generation_ES import entity_generation
from src.candidate_selection import candidate_selection
import pickle
import time
import configparser

def str2bool(string):
    '''
    Converts the user input of any 'True' like string to boolean

    :param string: string to be converted
    :return: boolean
    '''
    return str(string).lower() in ("yes", "true", "t", "1")


def parse_config_arguments():
    '''
    Reads all the config arguments

    :return: returns config file values.
    '''

    cfg_reader = configparser.ConfigParser()
    cfg_reader.read_file(open('config.ini'))
    config = cfg_reader['default']

    warc_bool = str2bool(config['process_warc'])
    es_bool = str2bool(config['perform_candidate_generation'])
    local_bool = str2bool(config['local_elasticsearch'])

    if warc_bool is False:
        filename_warcs = config['filename_warcs']
        warcs_to_process = None
    else:
        filename_warcs = None
        warcs_to_process = config['warc_archives']

    model_for_ranking = config['model_for_ranking']

    return warc_bool, es_bool, local_bool, filename_warcs, warcs_to_process, model_for_ranking


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

    warc_bool, es_bool, local_bool, filename_warcs, warcs_to_process, model_for_ranking = parse_config_arguments()

    if warc_bool:
        list_of_warcnames = start_processing_warcs(warcs_to_process)
    else:
        with open(filename_warcs) as f:
            list_of_warcnames = list(f.readlines())

    warc_texts = read_all_warcs(list_of_warcnames)

    if es_bool:
        candidate_dict = generate_and_save_entities(warc_texts, local_bool)
    else:
        with open("candidate_dictionary.pkl", "rb") as f:
            candidate_dict = pickle.load(f)

    for idx, warc in enumerate(warc_texts):
        output = disambiguate_entities(warc, candidate_dict, model_for_ranking)

        with open(f"results/annotations_{list_of_warcnames[idx][:-4]}_{model_for_ranking}", 'w') as outfile:
            for entity_tuple in output:
                outfile.write(entity_tuple[0] + '\t' + entity_tuple[1] + '\t' + entity_tuple[2] + '\n')
