import gzip
import sys
import requests
from html2text import html2text
import spacy
import pickle
from elasticsearch import Elasticsearch
import json
import trident
import nltk
from nltk.corpus import wordnet as wn
from textblob import TextBlob
from nltk.corpus import stopwords


KEYNAME = "WARC-Record-ID"
WARC_DIRECTORY = "data/warcs/"
WARC_FILE = "CC-MAIN-20200927121105-20200927151105-00583.warc.gz"
INPUT_FILE = "data/sample.warc.gz"
KBPATH='assets/wikidata-20200203-truthy-uri-tridentdb'

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

# def split_records(stream):
#     payload = []
#     for idx,line in enumerate(stream):
#         print(line.strip())
#         if "WARC-Target-URI" in line.strip():
#             line_split = line.split("URI: ")[1]
#             clean_line = line_split.split("\n")[0]
#             payload.append(clean_line)
#
#         if idx > 5000:
#             return payload
#
#     return payload

def split_records(stream):
    payload = ''
    for line in stream:
        if line.strip() == "WARC/1.0":
            yield payload
            payload = ''
        else:
            payload += line
    yield payload

def read_warc_files():

    with gzip.open(INPUT_FILE, 'rt', errors='ignore') as fo:
        webpage_urls = split_records(fo)
        #print(len(webpage_urls))
        #url = webpage_urls[0]
        url = "https://www.theguardian.com/sport/2021/nov/09/emma-raducanu-torben-beltz-tennis-coach-upper-austria-ladies-linz"
        r = requests.get(url,timeout=10)
        html_code = r.text
        clean_text = html2text(html_code)
        print(clean_text)
        doc = nlp(clean_text)
        doc_dict = []
        for ent in doc.ents:
            dict = {}
            dict["Entity"] = ent.text.strip()
            dict["Label"] = ent.label
            dict["Start"] = ent.start_char
            dict["End"] = ent.end_char
            doc_dict.append(dict)

        #with open('entity_lists/list_of_entities_gua_1.txt', 'wb') as fp:
        #    pickle.dump(doc_dict, fp)

        ## To read
        #with open('entity_lists/list_of_entities_nu_1.txt', 'rb') as fp:
        #    entity_list = pickle.load(fp)
        exit(1)

def search(query,size):
    e = Elasticsearch("http://fs0.das5.cs.vu.nl:10010/")
    p = { "query" : { "query_string" : { "query" : query } }, "size":size}
    response = e.search(index="wikidata_en", body=json.dumps(p))
    id_labels = []
    if response:
        for hit in response['hits']['hits']:
            print()
            print(hit)
            print()
            print("---------------------------------")
            label = hit['_source']['schema_name']
            id = hit['_id']
            #id_labels.setdefault(id, set()).add(label)
            id_labels.append(id)
    return id_labels

# def get_nouns_from_definition(synsets):
#     definitions = [x.definition() for x in synsets]
#     return [ [ent.text.strip() for ent in nlp(definition).ents] for definition in definitions]


## Using NLTK tokenization
def get_nouns_from_definition(synsets):
    definitions = [x.definition() for x in synsets]
    print(definitions)
    is_noun = lambda pos:pos[:2] == "NN"

    return [ [word for (word,pos) in nltk.pos_tag(nltk.word_tokenize(definition)) if word not in stop_words if is_noun] for definition in definitions]

## Using TextBlob tokenization.
# def get_nouns_from_definition(synsets):
#     definitions = [x.definition() for x in synsets]
#     print(definitions)
#     return [ TextBlob(definition).noun_phrases for definition in definitions]

def order_entities(list_of_uris):

    # First split on "/", then take the last part of the URI including the entity number.
    # Afterwards, remove the Q and the ">" from the entity number to get the number itself
    # Then convert everything to int so that it can be sorted according to the entity numbers

    entity_numbers = [ int( uri.split("/")[-1][1:][:-1] ) for uri in list_of_uris]

    return [x for _, x in sorted( zip (entity_numbers, list_of_uris  ) )]

def entity_linking():

    with open('entity_lists/list_of_entities_gua_1.txt', 'rb') as fp:
        entity_list = pickle.load(fp)

    for idx, entity in enumerate( entity_list ):

        if idx > 50:
            exit(1)

        #check_entity = entity["Entity"]
        check_entity = "Glasgow"
        # Impossible: USA/US
        print("++++++++++++++++++++")
        print("Checking for Entity: ", check_entity)

        # if any( x in check_entity for x in ["/", "[", "]", "(", ")", "%"] ):
        #     print("Faulty entity, skipping")
        #     print("++++++++++++++++++++")
        #     continue

        synsets = wn.synsets(check_entity, pos=wn.NOUN)[:5]

        if synsets:
            search_size = 10
            synonyms = list( set( [lemma.name().replace("_", " ") for x in synsets for lemma in x.lemmas()] ) )
            synonym_length = len(synonyms)

            if synonym_length <= 1:
                #print("NO SYNONYMS FOUND, CHECKING DEFINITION")
                synonyms = list( set( get_nouns_from_definition(synsets)[0] ) )
                print(synonyms)
                exit(1)
                search_size = 10

            elif 2 < synonym_length <= 5:
                #print("REDUCING SIZE!, TOO MANY SYNONYMS")
                search_size = 10

            elif synonym_length > 5:
                #print("TOO MANY SYNONYMS, ONLY CHECKING FIRST 10")
                synonyms = synonyms[:5]
                search_size = 10

            #synonyms += get_nouns_from_definition(synsets)[0][:5]

            list_of_uris = []
            for synonym in synonyms:
                print("++++++++++++++++++++")
                print("CHECKING FOR ENTITY: ", check_entity, " AND SYNONYM: ", synonym)
                list_of_uris += search(synonym, search_size)
                list_of_uris += search("(%s) AND (%s)" % (check_entity, synonym), search_size )
                print("++++++++++++++++++++")


        else:
            print("No Synonyms,querying normally")
            list_of_uris = search(check_entity,20)

        list_of_uris = list ( set ( list_of_uris)) # For only obtaining the unique ones

        ordered_uris = order_entities(list_of_uris)
        if ordered_uris:
            print("Entity: ", check_entity, " ;  Corresponding best URI: ", ordered_uris)
            print("++++++++++++++++++++")
            print()

        else:
            print("No page found for entity: ", check_entity, ".  Continuing search.")
            print("++++++++++++++++++++")
            print()

        exit(1)

#def link_lookup(url):

    # QUERIES FOR SPARQL, NOT NEEDED
    # query = "PREFIX wde: <http://www.wikidata.org/entity/> " \
    #         "PREFIX wdp: <http://www.wikidata.org/prop/direct/> " \
    #         "PREFIX wdpn: <http://www.wikidata.org/prop/direct-normalized/> " \
    #             "select ?s where { ?s wdp:P31 wde:Q145 . } LIMIT 10"

    # query = "PREFIX wde: <http://www.wikidata.org/entity/> " \
    #         "SELECT DISTINCT ?v WHERE { ?v ?p wde:Q145 } "

    # Load the KB
    #db = trident.Db(KBPATH)
    # url = "<http://www.wikidata.org/entity/Q145>"

    #results = db.sparql(query)

    #print(results)


    # print("URL: ", url)
    #
    # term_id = db.lookup_id(url)
    # print("Term id: ", term_id)
    # po = db.po(term_id) # returns relations, such as P31 or http://www.wikidata.org/prop/direct/P279
    # print(po)
    # relation = db.lookup_str(po[5][0])
    #
    # print(relation)
    #
    # exit(1)

#read_warc_files()

entity_linking()




