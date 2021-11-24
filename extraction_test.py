import os
import re
import gzip
import glob
import spacy
import pickle
import fasttext
import html5lib
from bs4 import BeautifulSoup
from multiprocessing import get_context


# loading the spacy language model
nlp = spacy.load('en_core_web_md')

# loading the language detection model
lang_det = fasttext.load_model('lid.176.ftz')
fasttext.FastText.eprint = lambda x: None

# define some constants regarding where the WARC record IDs can be found
# as well as which NER labels we are filtering for
KEYNAME = 'WARC-Record-ID'
TARGET_LABELS = {'GPE', 'LOC',
                 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART',
                 'LAW', 'FAC'}  # removed EVENT and NORP and LANGUAGE

EXCEPTIONS = {'WARC-Type', 'GMTCache-Control', 'User-AgentConnection', 'GTMContent-Type', 'ul li' '9px',"WARC-TARG", "h3"}
re_compile = lambda x: re.compile(f'^{x}.*$')

EXCEPTIONS = {re_compile(x) for x in EXCEPTIONS}
# print(EXCEPTIONS)

PUNCTUATION = {'!', '/', '%', '|', '\\', ']', '[', '^', '<', '{', '}', '~', '`', '(', ')',
               '"', '=', '>', ';', '@', '\'', '*', '+', '?', '_', '...', ',', '--', }
STR_PUNCTUATION = ''.join([punct for punct in PUNCTUATION])
# print(STR_PUNCTUATION)

counter_entities = 0


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
    # print(skip)

    payload_content = payload.splitlines()

    for line in payload_content:
        # if 'Content-Type: text/html' in line:
        if '<!DOCTYPE html>' in line:
        # if 'Content-Type: text/html' in line:
            skip = False
            break
    #     # if '<!DOCTYPE html>' in line:
    #     if 'text/html' in line:
    #         skip = False
    #         continue
        else:
            skip = True

    if skip is False:
        # for line in payload.splitlines():
        for line in payload_content:
            soup = BeautifulSoup(line, features='html5lib')
            text = soup.body.get_text(strip=True).strip()
            text = text.replace('\ufeff', '')
            # filter ascii control characters
            text = re.sub(r'[\x00-\x1F]+', '', text)
            # print(text.strip())
            # exit(1)
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
    global counter_entities

    entities = []
    # print(text)
    doc = nlp(text)
    # looping through the sentences in the text
    for sent in doc.sents:
        # identifying entities
        for ent in sent.ents:
            cleaned_mention = ent.text.strip(STR_PUNCTUATION)
            if not any(re.match(exception, cleaned_mention) for exception in EXCEPTIONS)\
                    and not any(punct in cleaned_mention[1:-1] for punct in PUNCTUATION)\
                    and ent.label_ in TARGET_LABELS:
                # if not any(punct in cleaned_mention[1:-1] for punct in PUNCTUATION):
                    # filter out ascii control characters
                cleaned_mention = re.sub(r'[\x00-\x1F]+', '', cleaned_mention)
                if cleaned_mention:
                    #print(cleaned_mention)
                    tuple_to_add = (cleaned_mention, ent.label_, sent.text)
                    entities.append(tuple_to_add)

    counter_entities += len(entities)
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
    global counter_entities
    basename = os.path.basename(archive_path).rstrip('.warc.gz')
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
                    # output_dict[key]['text'] = text
                        counter += 1
                        if counter % 10 == 0:
                            print(counter)

    print("Total number of entities: ", counter_entities)

    with open(f'outputs/{basename}_entities.pkl', 'wb') as outfile:
        pickle.dump(output_dict, outfile)


if __name__ == '__main__':

    # all_paths = glob.glob('data/warcs/**.gz')
    # processes = len(all_paths)
    #
    # with get_context('spawn').Pool(processes) as p:
    #     p.map(process_archive, all_paths)

    process_archive('data/sample.warc.gz')