import os
import gzip
import glob
import spacy
import string
import pickle
import subprocess
import fasttext
import html5lib
from bs4 import BeautifulSoup
from multiprocessing import Pool
from collections import defaultdict
from multiprocessing import get_context

# loading the spacy language model
try:
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'tok2vec'])
except OSError:
    # if the en_core_web_sm model is installed, install it
    subprocess.call('python3 -m spacy download en_core_web_sm', shell=True)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'tok2vec'])

# loading the language detection model
lang_det = fasttext.load_model('../lid.176.ftz')

# define some constants regarding where the WARC record IDs can be found
# as well as which NER labels we are filtering for
KEYNAME = 'WARC-Record-ID'
TARGET_LABELS = {'EVENT', 'GPE', 'LOC', 'NORP',
                 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART',
                 'LAW', 'LANGUAGE', 'FAC'}

# PUNCTUATION without dots, hyphens and apostrophes that might likely appear in named entities
PUNCTUATION = str({punct for punct in string.punctuation if punct != '.' and punct != '-' and punct != "'"})


def split_records(stream):
    """
    Splits a WARC archive into individual websites (records).

    :param stream: a WARC archive containing records
    :return: a generator object yielding individual websites (records)
    """
    payload = ''
    for line in stream:
        if line.strip() == "WARC/1.0":
            yield payload
            payload = ''
        else:
            payload += line
    yield payload


def filter_for_english_text(payload):
    """
    Finds content that contains a <!DOCTYPE html> tag
    and contains English text.

    :param payload: an instance of a WARC record
    :return: a string of text or None if nothing of that kind is found
    """
    for line in payload.splitlines():
        if '<!DOCTYPE html>' in line:
            soup = BeautifulSoup(line, features='html5lib')
            text = soup.body.get_text(strip=True).strip()
            text = text.replace('\ufeff', '')
            if text:
                try:
                    languages = lang_det.predict(text)
                except ValueError:
                    text = text.replace('\n', '')
                    languages = lang_det.predict(text)
                if '__label__en' in languages[0]:
                    return text
    return None


def collect_entities(text):
    """
    Finds named entities in a text and returns
    a list containing the entities if they both
    have the appropriate target labels and not contain
    equal signs and semicolons.

    :param text: a string of text
    :return: a list of named entities detected
    """
    entities = []
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in TARGET_LABELS:
            if not any(punct in ent.text[1:-1] for punct in PUNCTUATION):
                tuple_to_add = (ent.text.strip(PUNCTUATION), ent.label_)
                entities.append(tuple_to_add)
    return entities


def extract_key(payload):
    """
    Extracts the WARC-Record-ID from
    a payload.

    :param payload: an instance of a WARC record
    :return: an empty string if no payload is found or a key if it is
    """
    if payload == '':
        return
    key = None
    for line in payload.splitlines():
        if line.startswith(KEYNAME):
            key = line.split(': ')[1]
    return key


def process_payload(payload):
    """
    Attempts to find a string of English text
    in a WARC record, and if it finds one,
    it extracts the record key too.

    :param payload: an instance of a WARC record
    :return: a tuple of a record key and text
    """
    key = None
    text = filter_for_english_text(payload)
    if text:
        key = extract_key(payload)
    return key, text


def process_archive(archive_path):
    """
    Given a path to a WARC archive, it processes its records
    by attempting to extract a string of English text and
    identifying entities in it.

    :param archive_path: a filepath to the WARC archive
    :return: None, it writes out the entities in the outputs folder
    """
    basename = os.path.basename(archive_path).rstrip('.warc.gz')
    counter = 0
    output_dict = defaultdict(dict)
    with gzip.open(archive_path, 'rt', errors='ignore', encoding='utf8') as stream:
        payloads = split_records(stream)
        for payload in payloads:
            key, text = process_payload(payload)
            if key and text:
                try:
                    entities = collect_entities(text)
                except ValueError:
                    continue
                output_dict[key]['entities'] = entities
                output_dict[key]['text'] = text
                counter += 1
                if counter % 10 == 0:
                    print(counter)
    with open(f'../outputs/{basename}_entities.pkl', 'wb') as outfile:
        pickle.dump(output_dict, outfile)


if __name__ == '__main__':

    all_paths = glob.glob('../data/warcs/**.gz')
    processes = len(all_paths)

    with get_context('spawn').Pool(processes) as p:
        p.map(process_archive, all_paths)
