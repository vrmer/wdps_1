import gzip
import sys
import requests
import pickle
from elasticsearch import Elasticsearch
import json
import trident
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from copy import deepcopy
import string

KBPATH='assets/wikidata-20200203-truthy-uri-tridentdb'

stop_words = set(stopwords.words("english"))
stop_words.add("-")
lemmatizer = WordNetLemmatizer()

def search(query,size):
    """
    Performs Elastic search

    :param query: query string for Elastic Search process
    :param size: determines the number of URIs returned from Elastic Search
    :return: a list of URI's from the search process
    """
    # e = Elasticsearch("http://fs0.das5.cs.vu.nl:10010/")
    # e = Elasticsearch('http://localhost:9200')
    e = Elasticsearch(timeout=30)
    p = { "query" : { "query_string" : { "query" : query } }, "size":size}
    response = e.search(index="wikidata_en", body=json.dumps(p))
    id_labels = []
    if response:
        for hit in response['hits']['hits']:
            print()
            print(hit)
            print()
            print("---------------------------------")
            id = hit['_id']
            label = hit['_source']['schema_name']
            if "schema_description" in hit['_source']:
                description = hit['_source']['schema_description']
            else:
                description = ""
            rdfs_label = hit['_source']['rdfs_label']
            score = hit["_score"]
            uri_dict = {"uri": id, "score": score, "rdfs": rdfs_label, "name": label, "description": description}
            id_labels.append(uri_dict)
    return id_labels


def perform_similarity_algorithm(text, synsets):
    """
    Manual implementation of the Simplified Lex Algorithm.
    The idea is to match the lemmatized nouns from both the context and definition of the various synsets.
    Afterwards, rank them according to the number of similar occurences.

    :param text: the sentence in which the recognized entity occurs
    :param synsets: the synonym sets found by NLTK's wordnet
    :return: the top 3 synsets and best definition based on the similarity counts.
    """
    text_tok = nltk.word_tokenize(text)
    is_noun = lambda pos: pos[:2] == "NN"
    text_clean = [lemmatizer.lemmatize(word) for (word,pos) in nltk.pos_tag(text_tok) \
                  if word.strip() not in stop_words and word.strip() not in string.punctuation and not word.strip().isdigit() and is_noun]

    definitions = get_nouns_from_definition(synsets)
    list_of_counts = []

    for definition in definitions:
        similarity_count = sum( [1 if lemmatizer.lemmatize(word) in text_clean else 0 for word in definition])
        list_of_counts.append(similarity_count)

    best_synsets = order_list_from_list(synsets,list_of_counts, True)[:3]
    best_definition = order_list_from_list(definitions,list_of_counts, True)[0]

    return best_synsets, best_definition

def get_nouns_from_definition(synsets):
    """
    Returns a list of lists that contain the nouns for each synset definition

    :param synsets: the synonym sets found by NLTK's Wordnet
    :return: a list of lists containing nouns
    """
    definitions = [x.definition() for x in synsets]
    is_noun = lambda pos:pos[:2] == "NN"
    return [ [word for (word,pos) in nltk.pos_tag(nltk.word_tokenize(definition)) \
            if word.strip() not in stop_words and word.strip() not in string.punctuation and not word.strip().isdigit() and is_noun ] \
            for definition in definitions]

def sort_list_manually(to_sort, base):
    """
    Manual sorting algorithm for maintaining the order of the synsets found in NLTK's Wordnet while sorting the list
    in descending order

    :param to_sort: the list of synsets that needs to be sorted
    :param base: the list that provides the integers on which the to_sort list needs to be sorted
    :return: a sorted list of synsets
    """
    copy_base = deepcopy(base)

    while sum(copy_base) > 0:
        max_value = max(copy_base)
        copy_base.remove(max_value)
        to_sort.insert(0, to_sort.pop(base.index(max_value)))

    return to_sort

def order_list_from_list(to_sort, base, reverse):
    """
    Sort the list of synsets or entities in descending and ascending order respectively based on the base list.

    :param to_sort: the list of synsets that needs to be sorted
    :param base: the list that provides the integers on which the to_sort list needs to be sorted
    :param reverse: sort the list of synsets in descending order
    :return: a sorted list of synsets or entities
    """
    if reverse:
        return [x for x in sort_list_manually(to_sort,base)]
    else:
        return [x for _, x in sorted(zip(base, to_sort))]

def filter_uris(list_of_uris, entity):
    to_delete = []
    for idx,uri_dict in enumerate(list_of_uris):
        print(idx)
        if entity not in uri_dict["name"] and entity not in uri_dict["description"]:
            print(uri_dict)
            to_delete.append(idx)
        elif "Wikipedia disambiguation page" in uri_dict["description"]:
            to_delete.append(idx)

    return [x for idx,x in enumerate(list_of_uris) if idx not in to_delete]

def entity_generation(check_entity, context):
    """
    Performs the entity generation for all entities and corresponding texts

    :return: a sorted list of URI's for the trident database.
    """

    #check_entity = entity["Entity"]
    check_entity = "Washington"
    #text = "The United States of America (U.S.A. or USA), commonly known as the United States (U.S. or US) or America, is a country primarily located in North America."
    #text = "George Washington (February 22, 1732 – December 14, 1799) was an American military officer, statesman, and Founding Father who served as the first president of the United States from 1789 to 1797"
    #context = ""
    print("++++++++++++++++++++")
    print("Running ES for Entity: ", check_entity)

    synsets = wn.synsets(check_entity, pos=wn.NOUN)

    if synsets:
        search_size = 8
        synset_length = len(synsets)

        if synset_length <= 1:
            synonyms = list( set( get_nouns_from_definition(synsets)[0] ) )
        else:
            if not context:
                best_synsets = synsets[:3]
                best_definition = list( set( get_nouns_from_definition(synsets)[0] ) )
                search_size = 8
            else:
                best_synsets, best_definition = perform_similarity_algorithm(context, synsets)

            synonyms = [lemma.name().replace("_", " ") for x in best_synsets for lemma in x.lemmas()]
            synonyms = list( set( synonyms + best_definition ) )[:13]

        #synonyms = ['George Washington', '1st', 'President', 'United', 'States', 'commander-in-chief', 'Continental', 'Army', 'American', 'Revolution', '1732-1799']

        list_of_uris = []
        synonyms_iter = synonyms[:-1]

        for  synonym in synonyms:
            print("CHECKING FOR ENTITY: ", check_entity, " AND SYNONYM: ", synonym)

            list_es = search("(%s) AND (%s)" % (check_entity, synonym), search_size)
            if not list_es:
                list_of_uris += list_es
                continue
            else:
                list_of_uris += list_es + search(synonym, search_size)


            print("++++++++++++++++++++")

    else:
        print("No Synsets detected,querying normally")
        list_of_uris = search(check_entity,20)


    list_of_uris = [dict(t) for t in {tuple(d.items()) for d in list_of_uris}]
    print(len(list_of_uris))
    list_of_uris = filter_uris(list_of_uris, check_entity)
    print(len(list_of_uris))

    '''
    First split on "/", then take the last part of the URI including the entity number.
    Afterwards, remove the Q and the ">" from the entity number to get the number itself
    Then convert everything to int so that it can be sorted according to the entity numbers
    '''
    entity_numbers = [int(dictionary["uri"].split("/")[-1][1:][:-1]) for dictionary in list_of_uris]
    ordered_uris = order_list_from_list(list_of_uris, entity_numbers, False)

    print([dic["uri"] for dic in ordered_uris])

    if ordered_uris:
        print("Entity: ", check_entity, " ;  Corresponding best URI: ", ordered_uris)
        print("++++++++++++++++++++")
        print()

    else:
        print("No page found for entity: ", check_entity, ".  Continuing search.")
        print("++++++++++++++++++++")
        print()

    return ordered_uris

PKL_file = "outputs/CC-MAIN-20200927121105-20200927151105-00583_entities.pkl"

with open(PKL_file, "rb") as infile:
    texts = pickle.load(infile)

for key, entities in texts.items():
    for idx,entity_tuple in enumerate(entities):
        mention, label, context = entity_tuple
        list_of_uris = entity_generation("Washington", "George Washington (February 22, 1732 – December 14, 1799) was an American military officer, statesman, and Founding Father who served as the first president of the United States from 1789 to 1797")
        exit(1)

#entity_generation("Washington", "George Washington (February 22, 1732 – December 14, 1799) was an American military officer, statesman, and Founding Father who served as the first president of the United States from 1789 to 1797")
'Washington'