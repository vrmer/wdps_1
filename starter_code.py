from collections import defaultdict
import fasttext
from src.extraction import start_processing_warcs
from src.entity_generation_ES import entity_generation
from src.candidate_selection import candidate_selection
import argparse
import pickle
import sys
import time
from itertools import islice
from multiprocessing import Pool
from itertools import repeat
import multiprocessing

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

    cmd_parser.add_argument('-m', '--model_for_ranking', choices=['popularity','lesk','glove','bert'], required=False,
                            help="Required argument, write which model to use for candidate ranking. Possible options:"
                            "popularity | lesk | glove | bert, \n Default: popularity")

    cmd_parser.add_argument('-l', '--local', required=True,
                            help="Required argument, write 'True' if you want to run the local elastic search algorithm,"
                                 "'False' otherwise")

    cmd_parser.add_argument('-p', '--process_warcs', type=str, required= False,
                        help="Optional argument, write 'True' if you want to process the warc file(s), False otherwise")

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

    return warc_bool, es_bool, filename_warcs, parsed.model_for_ranking,local_bool


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


def split_mention_dict(mention_dict, slices):
    """
    Takes a dictionary containing mentions extracted
    from WARC records and splits into a number of slices.

    :param mention_dict: a dictionary containing mentions extracted from WARC records
    :param slices: a predetermined number of slices the mention_dict will be split into
    :return: a list containing the sliced mention dictionaries
    """
    sliced_mention_dicts = []
    slice_size = int(len(mention_dict) / slices)
    prev_idx = 0

    for i in range(0, len(mention_dict), slice_size):
        sliced_dict = {k: mention_dict[k] for k in islice(mention_dict, prev_idx, prev_idx + slice_size)}
        sliced_mention_dicts.append([sliced_dict])
        prev_idx += slice_size
    return sliced_mention_dicts


def merge_pooled_processes(pooled_processes):
    """
    Merges the outcome of pooled processes resulting from
    carrying out multiprocessing to query the Elasticsearch.

    :param pooled_processes: a list containing the output of pooled processes
    :return: a dictionary containing all candidate entities extracted by the processes
    """
    unique_uris = set()
    merged_processes = defaultdict(list)

    for pooled_process in pooled_processes:
        for target_entity, returned_queries in pooled_process.items():
            for returned_query in returned_queries:
                uri = returned_query['uri']
                if uri not in unique_uris:
                    unique_uris.add(uri)
                    merged_processes[target_entity].append(returned_query)

    return merged_processes


def generate_and_save_entities(warcs, slice_no, slices, local_bool):
    '''
    Generates and saves the entities found through elasticsearch.
    This process is parallelized for both the elasticsearch server and the current program.

    :param warcs: a list of all warc dictionaries (1 dict per warc file)
    :param slice_no: ??
    :param slices: ??
    :param local_bool: Indicates whether the elasticsearch should be run through a local or public client
    :return: a dict of all candidates
    '''

    dict_of_candidates = {}
    entities_checked = 0

    curr_proc = multiprocessing.current_process()
    print("Starting candidate search for process: ", curr_proc._identity[0])

    for warc in warcs:
        for key, entities in warc.items():
            for mention, label, context in entities:
                if mention not in dict_of_candidates.keys():
                    # print("Checking entity: ", mention)
                    list_of_uris = entity_generation(mention, context, slice_no, slices, local_bool)
                    dict_of_candidates[mention] = list_of_uris
                    entities_checked +=1
                    # print("Checked entity: ", mention)
                    # print("Best result: ", list_of_uris[0] if list_of_uris else "Empty")

                    if entities_checked % 200 ==0:
                        print("Entities checked: ", entities_checked)
                        if entities_checked % 400 == 0:
                            curr_proc = multiprocessing.current_process()
                            print("Saving Temp Candidate_Dict for process id: ", curr_proc._identity[0])
                            with open('outputs/candidate_dictionary_' + str(curr_proc._identity[0]) + '.pkl', 'wb') as f:
                                pickle.dump(dict_of_candidates,f)

    return dict_of_candidates

def disambiguate_entities( warc_texts, candidate_dict, method='popularity' ):

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
    slices = 5

    warc_bool, es_bool, fw, model, local_bool = parse_cmd_arguments()

    if warc_bool:
        list_of_warcnames = start_processing_warcs(lang_det)
    else:
        with open(fw) as f:
            list_of_warcnames = list(f.readlines())

    warc_texts = read_all_warcs(list_of_warcnames)

    subdicts = split_mention_dict(warc_texts[0], slices)
    slice_list = [slices]*slices

    if es_bool:
        pool = Pool(slices)
        pooled_processes = pool.starmap(generate_and_save_entities, zip(subdicts, range(slices), slice_list, repeat(local_bool)))

        merged_processes = merge_pooled_processes(pooled_processes)

        with open('outputs/candidate_dictionary.pkl', 'wb') as f:
            pickle.dump(merged_processes, f)
    else:
        with open("outputs/candidate_dictionary.pkl", "rb") as f:
            candidate_dict = pickle.load(f)

    for idx, warc in enumerate(warc_texts):
        output = disambiguate_entities(warc, candidate_dict, model)

        with open(f"results/annotations_' + {list_of_warcnames[idx][:-4]}", 'w') as outfile:
            for entity_tuple in output:
                outfile.write(entity_tuple[0] + '\t' + entity_tuple[1] + '\t' + entity_tuple[2] + '\n')

    # d = 'results'
    # sample_file = 'annotations_sample_entities'
    # files = os.listdir('results')
    # if sample_file in files:
    #     get_performance('data/sample_annotations.tsv',os.path.join(d, sample_file))
