import os
import gzip
import glob
import spacy
import pickle
import fasttext
import html5lib
from bs4 import BeautifulSoup
from multiprocessing import Pool
from collections import defaultdict

# from multiprocessing import set_start_method
# set_start_method("spawn")
from multiprocessing import get_context

nlp = spacy.load('en_core_web_sm', disable=['parser', 'tok2vec'])
lang_det = fasttext.load_model('lid.176.ftz')

KEYNAME = 'WARC-Record-ID'
TARGET_LABELS = {'EVENT', 'GPE', 'LOC', 'NORP', 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART'}


def split_records(stream):
    payload = ''
    for line in stream:
        if line.strip() == "WARC/1.0":
            yield payload
            payload = ''
        else:
            payload += line
    yield payload


def filter_for_english_text(payload):
    for line in payload.splitlines():
        if '<!DOCTYPE html>' in line:
            soup = BeautifulSoup(line, features='html5lib')
            text = soup.body.get_text(strip=True).strip()
            text = text.replace('\ufeff', '')
            # print(text)
            if text:
                try:
                    languages = lang_det.predict(text)
                except ValueError:
                    # print(repr(text))
                    text = text.replace('\n', '')
                    languages = lang_det.predict(text)
                # print(languages)
                if '__label__en' in languages[0]:
                    return text
    return None


def collect_entities(text):
    entities = []
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in TARGET_LABELS:
            if '=' not in ent.text and ';' not in ent.text:
                entities.append(ent.text)
    return entities


def extract_key(payload):
    if payload == '':
        return
    key = None
    for line in payload.splitlines():
        if line.startswith(KEYNAME):
            key = line.split(': ')[1]
    return key


def process_payload(payload):
    key = None
    text = filter_for_english_text(payload)
    if text:
        key = extract_key(payload)
    return key, text


def process_archive(archive_path):
    basename = os.path.basename(archive_path).rstrip('.warc.gz')
    counter = 0
    output_dict = defaultdict(dict)
    with gzip.open(archive_path, 'rt', errors='ignore', encoding='utf8') as stream:
        payloads = split_records(stream)
        for payload in payloads:
            key, text = process_payload(payload)
            # if counter == 2:
            #     break
            if key and text:
                try:
                    entities = collect_entities(text)
                except ValueError:
                    continue
                # print(key)
                output_dict[key]['entities'] = entities
                output_dict[key]['text'] = text
                # print(output_dict)
                # print('----------------')
                # print()
                counter += 1
                print(counter)
    with open(f'outputs/{basename}_entities.pkl', 'wb') as outfile:
        pickle.dump(output_dict, outfile)


# path = 'data/warcs/CC-MAIN-20200927121105-20200927151105-00583.warc.gz'


if __name__ == '__main__':

    # all_paths = glob.glob('data/warcs/**.gz')
    # processes = len(all_paths)
    processes = 1
    all_paths = ['data/warcs/CC-MAIN-20200929190110-20200929220110-00527.warc.gz']

    with get_context('spawn').Pool(processes) as p:
        p.map(process_archive, all_paths)
        # p.imap(process_archive, all_paths, chunksize=10)

    # with open('outputs/entities.pkl', 'wb') as outfile:
    #     pickle.dump(output_dict, outfile)
