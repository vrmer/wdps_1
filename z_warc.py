import gzip
import re
import warcio
from tqdm import tqdm
from html2text import HTML2Text, html2text
from bs4 import BeautifulSoup
import bs4
import requests
import fasttext
import spacy

lang_det = fasttext.load_model('lid.176.ftz')
nlp = spacy.load('en_core_web_sm')

# url = "https://www.theguardian.com/sport/2021/nov/09/emma-raducanu-torben-beltz-tennis-coach-upper-austria-ladies-linz"

# r = requests.get(url, timeout=10)

# html_code = r.text
# soup = BeautifulSoup(html_code, 'html.parser')

# print(type(soup.get_text()))

# print(clean_text.title.string)
# print(clean_text)

# for child in children:
#     if isinstance(child, bs4.element.Tag):
#         for item in child.strings:
#             print(item)
        # print(child)

# print(html_code)

# <title></title>
#

target_number = 50000
path = 'data/warcs/CC-MAIN-20200927121105-20200927151105-00583.warc.gz'


# def split_records(stream):
#     payload = ''
#     for line in stream:
#         if line.strip() == "WARC/1.0":
#             yield payload
#             payload = ''
#         else:
#             payload += line
#     yield payload

def split_records(stream):
    for idx, line in enumerate(stream):
        if '<!DOCTYPE html' in line:
            # clean = html2text(line)
            # print(line.strip())
            try:
                soup = BeautifulSoup(line, parser='html5lib')
                text = soup.body.get_text(strip=True)
                languages = lang_det.predict(text)
                if '__label__en' in languages[0]:
                    doc = nlp(text)
                    for ent in doc.ents:
                        print(ent.text)  # TODO: find all spacy labels and filter for the relevant ones
            # print(clean.strip())
                    print('-----------------------------------')
                    print()
            except:
                continue

# TODO: building only NER
# # Setting up the pipeline and entity recognizer.if model is not None:
#     nlp = spacy.load(model)  # load existing spacy model
#     print("Loaded model '%s'" % model)
# else:
#     nlp = spacy.blank('en')  # create blank Language class
#     print("Created blank 'en' model")if 'ner' not in nlp.pipe_names:
#     ner = nlp.create_pipe('ner')
#     nlp.add_pipe(ner)
# else:
#     ner = nlp.get_pipe('ner')


# TODO: filter for English with fasttext; use spacy
#  TODO: check whether filtering for DOCTYPE will work?


with gzip.open(path, 'rt', errors='ignore', encoding='utf8') as fo:
    split_records(fo)
    # for idx, record in enumerate(split_records(fo)):
    #     if idx == 15:
    #         break
        # else:
        #     # body = html2text(record)
        #     soup = BeautifulSoup(record, parser='html5lib')
        #     # children = soup.children
        #     # print(list(children))
        #     try:
        #         body = soup.body.get_text(strip=True)
        #         # body = soup
        #         print(body)
        #         print('------------------')
        #     except AttributeError or TypeError:
        #         continue


# with gzip.open(path, 'rt', errors='ignore', encoding='utf8') as fo:
#
#     with tqdm(total=target_number, desc='Gathering sites: ') as pbar:
#
#         htmls = []
#         site = ''
#
#         for idx, line in enumerate(fo):
#             if idx == target_number:
#                 break
#             else:
#                 if 'WARC/1.0' in line.strip():
#                     if 'Content-Type: text/html' in site:
#                         # print('Yes')
#                         htmls.append(site)
#                     site = ''
#                 else:
#                     site = site + line
#
#             pbar.update(1)
#
#
# # html_texter = HTML2Text()
# # html_texter.ignore_links = True
# # # html_texter.unicode_snob = True
# # html_texter.ignore_anchors = True
# # html_texter.images_to_alt = True
# # html_texter.single_line_break = True
# # html_texter.decode_errors = 'ignore'
#
# with tqdm(total=len(htmls), desc='Processing websites: ') as pbar:
#
#     bodies = []
#
#     for html in htmls:
#
#         soup = BeautifulSoup(html, parser='html5lib')
#
#         # body = soup.body
#         #
#         # print(body)
#
#         # if body:
#         #
#         #     clean = html_texter.handle(body)
#         #
#         #     bodies.append(clean)
#
#         try:
#             body = soup.body.get_text(strip=True)
#             # body = soup.body
#             bodies.append(body)
#             # lines = body.split('\n')
#             # print(lang_det.predict(lines, k=2)[0])
#         except TypeError or AttributeError:
#             continue
#
#         pbar.update(1)
#
# for item in bodies:
#     print('---------------------------------------------------')
#     print(item)

        # break

    # body_for_lang_id = body.replace('\n', ' ')
    # if '__label__en' in lang_det.predict(body_for_lang_id, k=2)[0]:
    #     doc = nlp(body)
    #     for ent in doc.ents:
    #         print(ent.text.strip())

# line_for_lang_id = line.replace('\n', ' ')
#                     if '__label__en' in lang_det.predict(line_for_lang_id, k=2)[0]:
