import gzip
import sys
import requests
from html2text import html2text
import spacy
import pickle
from elasticsearch import Elasticsearch
import json
import trident

KEYNAME = "WARC-Record-ID"
WARC_DIRECTORY = "data/warcs/"
WARC_FILE = "CC-MAIN-20200927121105-20200927151105-00583.warc.gz"
INPUT_FILE = "data/sample.warc.gz"
KBPATH='assets/wikidata-20200203-truthy-uri-tridentdb'

nlp = spacy.load("en_core_web_sm")

def split_records(stream):
    payload = []
    for idx,line in enumerate(stream):
        print(line.strip())
        if "WARC-Target-URI" in line.strip():
            line_split = line.split("URI: ")[1]
            clean_line = line_split.split("\n")[0]
            payload.append(clean_line)

        if idx > 5000:
            return payload

    return payload

def read_warc_files():

    with gzip.open(INPUT_FILE, 'rt', errors='ignore') as fo:
        webpage_urls = split_records(fo)
        #print(len(webpage_urls))
        #url = webpage_urls[0]
        url = "https://www.theguardian.com/sport/2021/nov/09/emma-raducanu-torben-beltz-tennis-coach-upper-austria-ladies-linz"
        r = requests.get(url,timeout=10)
        html_code = r.text
        clean_text = html2text(html_code)
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

def search(query):
    e = Elasticsearch("http://fs0.das5.cs.vu.nl:10010/")
    p = { "query" : { "query_string" : { "query" : query }}}
    response = e.search(index="wikidata_en", body=json.dumps(p))
    #id_labels = {}
    id_labels = []
    if response:
        for hit in response['hits']['hits']:
            label = hit['_source']['schema_name']
            id = hit['_id']
            #id_labels.setdefault(id, set()).add(label)
            id_labels.append(id)
    return id_labels

def link_lookup(url, entity):

    entity_page = url.split("/")[-1][:-1]

    # query = "PREFIX wde: <http://www.wikidata.org/entity/> " \
    #         "PREFIX wdp: <http://www.wikidata.org/prop/direct/> " \
    #         "PREFIX wdpn: <http://www.wikidata.org/prop/direct-normalized/> " \
    #             "select ?s where { ?s wdp:P31 wde:" + entity_page + " . } LIMIT 10"

    #query = "PREFIX wde: <http://www.wikidata.org/entity/" + entity_page + "> " \
    #        "SELECT DISTINCT ?v WHERE { ?v ?p wde:"+ entity + " . } "

    query = '''PREFIX wde: <http://www.wikidata.org/entity/''' + entity_page + '''> '''\
            '''SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }'''


    # Load the KB
    db = trident.Db(KBPATH)
    url = "<https://www.wikidata.org/wiki/Q145>"
    print("Looking for Entity Number: ", entity_page)
    print("URL: ", url)
    # results = db.sparql(query)
    term_id = db.search_id(url)
    print("Term id: ", term_id)
    indegree = db.indegree(term_id)


    print(indegree)
    exit(1)


def entity_linking():

    with open('entity_lists/list_of_entities_gua_1.txt', 'rb') as fp:
        entity_list = pickle.load(fp)

    for entity in entity_list:
        print("Checking for Entity: ", entity["Entity"])
        list_of_urls = search(entity["Entity"])

        if not list_of_urls:
            print("No urls found in the wikidata, skipping entity: ", entity["Entity"])
            continue

        for url in list_of_urls:
            print(list_of_urls)
            link_lookup(url, entity["Entity"])

        exit(1)

#read_warc_files()

entity_linking()




