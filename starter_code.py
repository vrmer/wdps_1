from functools import partial

import fasttext

from src.extraction import start_processing_warcs
from src.entity_generation_ES import entity_generation
# from src.context_vectors_2 import get_similarity_scores
import argparse
import pickle
import sys
import time
from itertools import islice
from multiprocessing import Pool
from elasticsearch import Elasticsearch

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


def split_entity_dict(entity_dict, slices=3):
    """

    :param entity_dict:
    :param slices:
    :return:
    """
    sliced_entity_dicts = []
    slice_size = int(len(entity_dict)/slices)
    prev_idx = 0
    # for i in range(0, len(entity_dict), slice_size):
    for i in range(0, len(entity_dict), slice_size):
        print(i)
        sliced_dict = {k: entity_dict[k] for k in islice(entity_dict, prev_idx, prev_idx+slice_size)}
        sliced_entity_dicts.append([sliced_dict])
        prev_idx += slice_size
    return sliced_entity_dicts


def read_all_es_results(list_of_names):
    list_of_candidates = []
    for file in list_of_names:
        with open("outputs/" + file, "rb") as infile:
            candidates = pickle.load(infile)
            list_of_candidates.append(candidates)

    return list_of_candidates

def generate_and_save_entities(warcs, slice_no, slices):
    dict_of_candidates = {}

    start = time.time()
    idx = 0

    # print(warcs)

    for warc in warcs:
        # print(warc)
        # print(type(warc))
        # exit(1)
        for key, entities in warc.items():
            for mention, label, context in entities:
                if mention not in dict_of_candidates.keys():
                    list_of_uris = entity_generation(mention, context, slice_no, slices)
                    dict_of_candidates[mention] = list_of_uris
                    print(f"{slice_no}\tEntity search completed for: ", mention)
                    print("Best Result:", list_of_uris if not list_of_uris else list_of_uris[0])
                    # exit(1)
                    if idx > 200:
                        print(time.time()-start)
                        exit(1)
                    else:
                        idx +=1


    with open('outputs/candidate_dictionary.pkl', 'wb') as f:
        pickle.dump(dict_of_candidates,f)

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

    # print(type(warc_texts[0]))
    # print(len(warc_texts[0]))

    subdicts = split_entity_dict(warc_texts[0], slices)
    slice_list = [slices]*slices
    # print(len(subdicts))
    # exit(1)

    # print(len(subdicts[0][0]))
    # for i, j in subdicts[0][0].items():
    #     print(i, j)
    #     break

    # exit(1)
    # subdicts = split_entity_dict(warc_texts[0], slices)

    subdicts = [
        [{'1': [('Washington', 'ORG', 'This is Washington.')]}],
        [{'2': [('Adams', 'PER', 'This is an Adams.')]}],
        [{'3': [('Budapest', 'LOC', 'Budapest is a great city.')]}]
    ]

    if es_bool:
        # e = Elasticsearch("http://fs0.das5.cs.vu.nl:10010/", timeout=30)
        # e = Elasticsearch("http://fs0.das5.cs.vu.nl:10010/", timeout=30)
        # candidate_dict = generate_and_save_entities(warc_texts, e)
        # candidate_dict = generate_and_save_entities(subdicts[0])

        pool = Pool(slices)
        # use_elasticsearch = partial(generate_and_save_entities)
        # candidate_dict = use_elasticsearch(subdicts)
        # pool = Pool(1)
        # pool.map(use_elasticsearch, warc_texts)
        entities = pool.starmap(generate_and_save_entities, zip(subdicts, range(slices), slice_list))
        print()
        for entity in entities:
            print(entity)
        # pool.map(generate_and_save_entities, subdicts)
        # exit(1)
        with open('outputs/test_dict.pkl', 'wb') as f:
            pickle.dump(entities, f)
        exit(1)
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

