import pickle
from elasticsearch import Elasticsearch
import json
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
punctuation = ['!', '/', '%', '|', '\\', ']', '[', '^', '<', '{', '}', '~', '`', '(', ')',
               '"', '=', '>', ';', '@', '\'', '*', '+', '?', '_', '...', ',', '--', ':']

def search(query,size):
    """
    Performs Elastic search

    :param query: query string for Elastic Search process
    :param size: determines the number of URIs returned from Elastic Search
    :return: a list of URI's from the search process
    """
    e = Elasticsearch("http://fs0.das5.cs.vu.nl:10010/", timeout = 30)
    #e = Elasticsearch('http://localhost:9200')
    #e = Elasticsearch(timeout=30)
    p = { "query" : { "query_string" : { "query" : query } }, "size":size}
    response = e.search(index="wikidata_en", body=json.dumps(p))
    id_labels = []
    if response:
        for hit in response['hits']['hits']:
            id = hit['_id']

            if "rdfs_label" not in hit['_source']:
                continue

            rdfs_label = hit['_source']['rdfs_label']
            name = hit["_source"]["schema_name"] if "schema_name" in hit["_source"] else ""
            description = hit['_source']['schema_description'] if "schema_description" in hit['_source'] else ""
            uri_dict = {"uri": id, "rdfs": rdfs_label, "name": name, "description": description}
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
                  if word.strip() not in stop_words and word.strip() not in punctuation and not word.strip().isdigit() and is_noun]

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
            if word.strip() not in stop_words and word.strip() not in punctuation and not word.strip().isdigit() and is_noun ] \
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
        return [x for _, x in sorted(zip(base, to_sort), key = lambda pair: pair[0])]

def filter_uris(list_of_uris, entity):
    """
    Filter the uris based on the content of the 'schema_description' and on the inclusion of the entity within either
    schema_name or schema_description. Remove uri through a list comprehension at the end

    :param list_of_uris: the list of dictionaries containing information per found entity
    :param entity: the entity for which the search was performed
    :return: a filtered list of dictionaries containing the uri's for the searched entity.
    """

    to_delete = []
    for idx,uri_dict in enumerate(list_of_uris):
        if entity not in uri_dict["name"] and entity not in uri_dict["description"]:
            to_delete.append(idx)
        elif "Wikipedia disambiguation page" in uri_dict["description"]:
            to_delete.append(idx)

    return [x for idx,x in enumerate(list_of_uris) if idx not in to_delete]

def entity_generation(check_entity, context):
    """
    Performs the entity generation for all entities and corresponding texts

    :return: a sorted list of URI's for the trident database.
    """

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
            synonyms = list( set( synonyms + best_definition ) )[:8]

        list_of_uris = []

        for synonym in synonyms:
            list_es = search("(%s) AND (%s)" % (check_entity, synonym), search_size)
            list_of_uris += list_es

            if not list_es:
                continue
            else:
                list_of_uris += search(synonym, search_size)

    else:
        #print("No Synsets detected,querying normally")
        list_of_uris = search(check_entity,20)

    #Find all unique dictionaries in the list and filter the URI
    list_of_uris = [dict(t) for t in {tuple(d.items()) for d in list_of_uris}]
    list_of_uris = filter_uris(list_of_uris, check_entity)

    '''
    First split on "/", then take the last part of the URI including the entity number.
    Afterwards, remove the Q and the ">" from the entity number to get the number itself
    Then convert everything to int so that it can be sorted according to the entity numbers
    '''
    entity_numbers = [int(dictionary["uri"].split("/")[-1][1:][:-1]) for dictionary in list_of_uris]
    ordered_uris = order_list_from_list(list_of_uris, entity_numbers, False)

    if not ordered_uris:
        ordered_uris = []

    return ordered_uris
