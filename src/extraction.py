import os
import re
import glob
import gzip
import spacy
import pickle
import fasttext
import html5lib
from bs4 import BeautifulSoup
from multiprocessing import get_context


# import language detector
fasttext.FastText.eprint = lambda x: None
lang_det = fasttext.load_model('lid.176.ftz')

# loading the spacy language model
nlp = spacy.load('en_core_web_md')

# define some constants regarding where the WARC record IDs can be found
# as well as which NER labels we are filtering for
KEYNAME = 'WARC-Record-ID'
TARGET_LABELS = {'GPE', 'LOC',
                 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART',
                 'LAW', 'FAC'}  # TODO: removed EVENT and NORP and LANGUAGE

# exceptions we decided to exclude due to their frequency and lack of relevance
EXCEPTIONS = {'WARC-Type', 'GMTCache-Control', 'User-AgentConnection', 'GTMContent-Type', 'ul li' '9px',"WARC-Targ", "h3", "WARC-Target"}
re_compile = lambda x: re.compile(f'(^)?{x}.*$')
EXCEPTIONS = {re_compile(x) for x in EXCEPTIONS}

# punctuation that we exclude when attempting to find entities
PUNCTUATION = {'!', '/', '%', '|', '\\', ']', '[', '^', '<', '{', '}', '~', '`', '(', ')',
               '"', '=', '>', ';', '@', '\'', '*', '+', '?', '_', '...', ',', '--', ':'}
STR_PUNCTUATION = ''.join([punct for punct in PUNCTUATION])

# list_of_filenames = []


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
    out_text = ''
    skip = True

    payload_content = payload.splitlines()

    for line in payload_content:
        if '<!DOCTYPE html>' in line:
            skip = False
            break
        else:
            skip = True

    if skip is False:
        for line in payload_content:
            soup = BeautifulSoup(line, features='html5lib')
            text = soup.body.get_text(strip=True).strip()
            text = text.replace('\ufeff', '')
            # filter ascii control characters
            text = re.sub(r'[\x00-\x1F]+', '', text)
            if text:
                try:
                    languages = lang_det.predict(text)
                except ValueError:
                    text = text.replace('\n', '')
                    languages = lang_det.predict(text)
                if '__label__en' in languages[0]:
                    out_text += text
        return out_text


def collect_entities(text):
    """
    Finds named entities in a text and returns
    the tuples containing the named entities, their
    labels, and the sentence they appear in.

    :param text: a string of text
    :return: a list of named entities detected
    """
    entities = []
    doc = nlp(text)
    # looping through the sentences in the text
    for sent in doc.sents:
        # identifying entities
        for ent in sent.ents:
            cleaned_mention = ent.text.strip(STR_PUNCTUATION)
            if not any(re.match(exception, cleaned_mention) for exception in EXCEPTIONS)\
                    and not any(punct in cleaned_mention[1:-1] for punct in PUNCTUATION)\
                    and ent.label_ in TARGET_LABELS:
                    # filter out ascii control characters
                cleaned_mention = re.sub(r'[\x00-\x1F]+', '', cleaned_mention)
                if cleaned_mention:
                    tuple_to_add = (cleaned_mention, ent.label_, sent.text)
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
    # list_of_filenames.append(basename + "_entities.pkl")
    counter = 0
    output_dict = dict()
    with gzip.open(archive_path, 'rt', errors='ignore', encoding='utf8') as stream:
        payloads = split_records(stream)
        for payload in payloads:
            if payload.strip():
                key, text = process_payload(payload)
                if key and text:
                    try:
                        entities = collect_entities(text)
                    except ValueError:
                        continue
                    if entities:
                        output_dict[key] = entities
                        counter += 1
                        if counter % 10 == 0:
                            print(counter)
                            break

    print(len(output_dict))
    with open(f'outputs/{basename}_entities.pkl', 'wb') as outfile:
        pickle.dump(output_dict, outfile)
    return basename + "_entities.pkl"


def start_processing_warcs(file_path):
    """

    :param lang_det:
    :param file_path:
    :return:
    """
    # list_of_filenames = []

    file_paths = glob.glob(file_path)
    print(file_paths)
    processes = len(file_paths)
    print(processes)

    with get_context('spawn').Pool(processes) as p:
        list_of_filenames = p.map(process_archive, file_paths)

    # for fp in file_paths:
    #     filename = process_archive(fp, lang_det)
    #     list_of_filenames.append(filename)

    # TODO: we should allow to provide filepath here
    # process_archive('data/sample.warc.gz', lang_det)

    with open("warc_file_names.txt", mode='wt', encoding='utf-8') as f:
        f.write('\n'.join(list_of_filenames))

    return list_of_filenames

    # all_paths = glob.glob('data/warcs/**.gz')
    # processes = len(all_paths)
    #
    # with get_context('spawn').Pool(processes) as p:
    #     p.map(process_archive, all_paths)


if __name__ == '__main__':

    # process_archive('data/sample.warc.gz', lang_det)
    start_processing_warcs('data/warcs/**.gz')
