from src.extraction import start_processing_warcs
from src.entity_generation_ES_no_parallel import entity_generation
from src.entity_generation_ES_no_parallel  import order_list_from_list
from src.candidate_selection import candidate_selection
# from src.score import get_performance
import argparse
import pickle
import sys
import time

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
stop_words.add("-")
lemmatizer = WordNetLemmatizer()
punctuation = ['!', '/', '%', '|', '\\', ']', '[', '^', '<', '{', '}', '~', '`', '(', ')',
               '"', '=', '>', ';', '@', '\'', '*', '+', '?', '_', '...', ',', '--', ':']

def str2bool(v):
  return str(v).lower() in ("yes", "true", "t", "1")

def parse_cmd_arguments():

    cmd_parser =argparse.ArgumentParser(description='Parser for Entity Linking Program')

    cmd_parser.add_argument('-s', '--save_es_results', type=str,
                        help="Required argument, write 'True' if you want to process and save the candidates "
                             "of the entity generation, 'False' otherwise")

    cmd_parser.add_argument('-p', '--process_warcs', type=str, required= False,
                        help="Optional argument, write 'True' if you want to process the warc file(s), False otherwise")

    cmd_parser.add_argument('-fp', '--filename_warcs', required='-p' in sys.argv,
                                   help="Required if -p == False, create a txt with the names of all the WARC pickle files, "
                                        "seperated by a '\\n' that need to be imported in the program")

    cmd_parser.add_argument('-m', '--model_for_ranking', choices=['popularity','lesk','glove','bert'], required=True,
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

def generate_and_save_entities(warcs):

    print("Starting Entity Generation Process")

    dict_of_candidates = {}
    entities_checked = 0

    for warc in warcs:
        for key, entities in warc.items():
            for mention, label, context in entities:
                if mention not in dict_of_candidates.keys():
                    print("checking entity: ", mention)
                    list_of_uris = entity_generation(mention, context)
                    dict_of_candidates[mention] = list_of_uris
                    entities_checked +=1

                    if entities_checked % 200 ==0:
                        print("Entities checked: ", entities_checked)
                        if entities_checked % 1000 == 0:
                            print("Saving Temp Candidate_Dict")
                            with open('../outputs/candidate_dictionary.pkl', 'wb') as f:
                                pickle.dump(dict_of_candidates,f)


    with open('../outputs/candidate_dictionary.pkl', 'wb') as f:
        pickle.dump(dict_of_candidates,f)

    return dict_of_candidates

def disambiguate_entities( warc_texts, candidate_dict,method='popularity' ):

    start = time.perf_counter()

    output = candidate_selection(warc_texts,candidate_dict,method)

    end = time.perf_counter()

    total_time = end - start
    print(f'Total time spent in disambiguating: {total_time}')

    return output

def extract_nouns_schemas(list_of_schemas):
    is_noun = lambda pos:pos[:2] == "NN"
    return [ [word for (word,pos) in nltk.pos_tag(nltk.word_tokenize(schema)) \
            if word.strip() not in stop_words and word.strip() not in punctuation and not word.strip().isdigit() and is_noun ] \
            for schema in list_of_schemas]

def find_best_match(clean_context, list_of_schemas, list_of_dicts):
    list_of_counts = []

    for schema in list_of_schemas:
        similarity_count = sum( [1 if lemmatizer.lemmatize(word) in clean_context else 0 for word in schema])
        list_of_counts.append(similarity_count)

    return order_list_from_list(list_of_dicts, list_of_counts, True)[0]


def run_lesk_algorithm(warcs, candidates, warc_names):

    for idx,warc in enumerate(warcs):
        list_of_results = []
        for key, entities in warc.items():
            for mention, label, context in entities:
                candidate_list = candidates[mention]
                if not candidate_list:
                    list_of_results.append( (key, mention, None) )

                schema_list = [x["description"] for x in candidate_list]
                clean_schema_list = extract_nouns_schemas(schema_list)
                clean_context = extract_nouns_schemas([context])
                best_uri = find_best_match(clean_context, clean_schema_list, candidate_list)
                list_of_results.append( (key, mention, best_uri) )

        with open('results/annotations_' + warc_names[idx], 'wb') as f:
            pickle.dump(list_of_results, f)

        break


if __name__ == '__main__':

    warc_bool, es_bool, fw, model = parse_cmd_arguments()

    if warc_bool:
        list_of_warcnames = start_processing_warcs()
    else:
        with open(fw) as f:
            list_of_warcnames = list(f.readlines())

    warc_texts = read_all_warcs(list_of_warcnames)

    if es_bool:
        candidate_dict = generate_and_save_entities(warc_texts)
    else:
        with open("../outputs/candidate_dictionary.pkl", "rb") as f:
            candidate_dict = pickle.load(f)

    # if model == "lesk":
    #     run_lesk_algorithm(warc_texts, candidate_dict, list_of_warcnames)
    #
    # else:
    #     pass
    #     for idx, warc in enumerate(warc_texts):
    #         output = disambiguate_entities(warc, candidate_dict, model)
    #
    #         with open(f"results/annotations_' + {list_of_warcnames[idx][:-4]}", 'w') as outfile:
    #             for entity_tuple in output:
    #                 outfile.write(entity_tuple[0] + '\t' + entity_tuple[1] + '\t' + entity_tuple[2] + '\n')

    # d = 'results'
    # sample_file = 'annotations_sample_entities'
    # files = os.listdir('results')
    # if sample_file in files:
    #     get_performance('data/sample_annotations.tsv',os.path.join(d, sample_file))

