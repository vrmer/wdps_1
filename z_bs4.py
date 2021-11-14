import gzip
import re
from bs4 import BeautifulSoup
import fasttext
import spacy
import html5lib
from tqdm import tqdm


# load models
lang_det = fasttext.load_model('lid.176.ftz')
nlp = spacy.load('en_core_web_sm')
# nlp = spacy.load('en_core_web_lg')

# path
path = 'data/warcs/CC-MAIN-20200927121105-20200927151105-00583.warc.gz'

# filter
target_labels = {'EVENT', 'GPE', 'LOC', 'NORP', 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART'}
target = 50_000

regex_pattern = re.compile('(^WARC-.*)|(^(X-Crawler-)?Content-.*)')
# javascript = {'=', ';'}

patterns = set()

with open('patterns.txt') as infile:
    lines = infile.readlines()
    for line in lines:
        pattern = line.strip()
        patterns.add(pattern)


def collect_entities(line):
    """

    :param line:
    :return:
    """
    entities = []
    doc = nlp(line)
    for ent in doc.ents:
        if ent.label_ in target_labels:
            # print(ent.text)
            if '=' not in ent.text and ';' not in ent.text:
                entities.append(ent.text)

    return entities


# variable to control whether lines are processed or skipped
skip = False
all_entities = []

with gzip.open(path, 'rt', errors='ignore', encoding='utf8') as stream:
    # with tqdm(total=target, desc='Iterating: ') as pbar:
    for idx, line in enumerate(stream):
        # print(idx)

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
                    if soup.style:
                        continue
                    elif soup.script:
                        continue
                    elif soup.body:
                        try:
                            text = soup.body.get_text()
                            try:
                                languages = lang_det.predict(text)
                                # stuff = f'{idx} + {languages}'
                                # print(stuff)
                                # print(languages)
                                if '__label__en' in languages[0]:
                                    # if len(text.strip()) >= 4 and ' ' in text and '||' not in text:
                                    if len(text.strip()) >= 4:
                                        if text.strip() not in patterns:
                                            print(text)
                                    # entity_list = collect_entities(text)
                                    # if entity_list:
                                        # print(entity_list)
                                        # all_entities.append(entity_list)
                            except:
                                continue
                        except:
                            continue

            # pbar.update(1)

            # if idx == target:
            #     with open('entities.txt', 'w') as outfile:
            #         for ent_list in all_entities:
            #             for ent in ent_list:
            #                 outfile.write(ent)
            #     break

                    # text = soup.get_text()
                    # print(text)
                    # body = soup.body
                    # children = body.children
                    # for child in children:
                    #     print(child)
                    # try:
                    #     for item in body.strings:
                    #         languages = lang_det.predict(item.strip())
                    #         if '__label__en' in languages[0]:
                    #             try:
                    #                 entity_list = collect_entities(item)
                    #                 if entity_list:
                    #                     print(entity_list)
                    #             except:
                    #                 continue
                    # except AttributeError:
                    #     continue



