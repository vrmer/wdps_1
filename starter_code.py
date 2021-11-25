from collections import defaultdict
import fasttext
from src.extraction import start_processing_warcs
from src.entity_generation_ES import entity_generation
import argparse
import pickle
import sys
import time
from itertools import islice
from multiprocessing import Pool


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

    return warc_bool, es_bool, filename_warcs


def read_all_warcs(list_of_warcs):
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


def read_all_es_results(list_of_names):
    list_of_candidates = []
    for file in list_of_names:
        with open("outputs/" + file, "rb") as infile:
            candidates = pickle.load(infile)
            list_of_candidates.append(candidates)

    return list_of_candidates


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


def generate_and_save_entities(warcs, slice_no, slices):
    dict_of_candidates = {}

    start = time.time()
    idx = 0

    for warc in warcs:
        for key, entities in warc.items():
            for mention, label, context in entities:
                if mention not in dict_of_candidates.keys():
                    list_of_uris = entity_generation(mention, context, slice_no, slices)
                    dict_of_candidates[mention] = list_of_uris
                    print("Entity search completed for: ", mention)
                    print("Best Result:", list_of_uris if not list_of_uris else list_of_uris[0])
                    if idx > 200:
                        print(time.time()-start)
                        break
                    else:
                        idx +=1
                    break
                break

    return dict_of_candidates


if __name__ == '__main__':

    lang_det = fasttext.load_model('lid.176.ftz')
    slices = 5

    warc_bool, es_bool, fw = parse_cmd_arguments()

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
        pooled_processes = pool.starmap(generate_and_save_entities, zip(subdicts, range(slices), slice_list))

        merged_processes = merge_pooled_processes(pooled_processes)

        with open('outputs/candidate_dictionary.pkl', 'wb') as f:
            pickle.dump(merged_processes, f)
    else:
        with open("outputs/candidate_dictionary.pkl", "rb") as f:
            candidate_dict = pickle.load(f)
