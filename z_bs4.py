import gzip
import re
from bs4 import BeautifulSoup
import fasttext
import spacy
import html5lib


# load models
lang_det = fasttext.load_model('lid.176.ftz')
nlp = spacy.load('en_core_web_sm')

# path
path = 'data/warcs/CC-MAIN-20200927121105-20200927151105-00583.warc.gz'

# filter
target_labels = {'EVENT', 'GPE', 'LOC', 'NORP', 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART'}

regex_pattern = re.compile('(^WARC-.*)|(^Content-.*)')


def collect_entities(line):
    """

    :param line:
    :return:
    """
    entities = []
    doc = nlp(line)
    for ent in doc.ents:
        if ent.label_ in target_labels:
            match = re.match('\p{IsCyrillic}', ent.label_)
            if not match:
                entities.append(ent.text)

    return entities


# variable to control whether lines are processed or skipped
skip = False

with gzip.open(path, 'rt', errors='ignore', encoding='utf8') as stream:
    for idx, line in enumerate(stream):

        # skip from True to False
        if skip is True:
            if 'WARC/1.0' in line:
                skip = False
            elif 'Content-Type: text/html' in line:
                skip = False
            else:
                pass

        else:
            # skip from False to True
            if 'Content-Type: ' in line and 'text/html' not in line:
                skip = True
            else:
                # try to skip lines associated with the crawl itself
                match = re.search(regex_pattern, line)
                if match:
                    continue
                else:
                    # process candidate results
                    soup = BeautifulSoup(line, features='html5lib')
                    body = soup.body
                    for item in body.strings:
                        languages = lang_det.predict(item.strip())
                        if '__label__en' in languages[0]:
                            try:
                                entity_list = collect_entities(item)
                                if entity_list:
                                    print(entity_list)
                            except:
                                continue



