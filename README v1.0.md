# README Group 3 Assignment 1

This repository contains the solution of group 3 for the Entity Linking Assignment from the course Web Processing Data Systems 2021 at VU Amsterdam. Our solution recognizes named entity mentions in Web pages and link them to Wikidata. In the following sections, the installation instructions and coding choices will be further elaborated. 

### Processing WARC files

The processing of the WARC files takes place in a parallel manner. The aim is to collect all records that contain text that is in English and that has recognizable entities in them ('mentions'). This means CSS and Javascript content has to be filtered out, as well as websites in languages aside from English. As output, we create one dictionary for each WARC archive. The dictionary keys are the WARC ids, and the value is a list of tuples in the <mention_name, mention_label, mention_context> format.

## Installation

To simplify the process of installing all required packaged, simply run in the command line:
`pip3 install -r requirements.txt`

In order to run the Spacy, it needs the English model, so download that through these means:
`python3 -m spacy download en_core_web_sm`

In order to run the Candidate Generation process, some corpora and data need to be downloaded for the NLTK package. To simplify this process, you can simply run the following line within the docker image to download all of the required data (NOTE: NLTK needed to properly run this process) : 
`$ python3 ./import_wordnet.py`

### Packages

General packages to add to requirements.txt:
- spacy
- fasttext 	(Needs Microsoft Visual C++ 14.0 or greater)
- html5lib
- bs4

For candidate generation specifically:
- nltk

## How to run the code

- Add which command(s) should be used to run the code 

## Coding Choices

In this section, we explain how we approach the challenges of candidate generation and disambiguation and motivate our choices.

As a first step, the WARC archives are unzipped, and they are split into individual records (websites). Since entity mentions can only be found in the HTML part of websites, we only process the record if it contains the <!DOCTYPE html> tag. If it does, we use BeautifulSoup to get all text from the body of the website. We use the html2text library for further cleaning of the text, namely the removal of HTML tags. If some text is found in the website body, we use fasttext to carry out language prediction, only keeping it if the text is English.

The next step is to use the spaCy library for sentence splitting and mention extraction. spaCy produces some state-of-the-art results in named entity recognition, which is why we chose it. For each sentence (which we assume to be the mention context), spaCy identifies potential entities with a label. We chose to retain only a subset of entities after carefully considering the outputs. The list of entity labels we retain is the following:

EVENT, GPE, LOC, NORP, ORG, PERSON, PRODUCT, WORK_OF_ART, LAW, LANGUAGE, FAC

Furthermore, we filter out mentions that are identified by spaCy but contain characters real-life entities do not normally contain. These include especially semicolons, equal signs, and brackets, that from observing the outputs we conclude often come from Javascript snippets that we fail to filter out. The remaining mentions are then added to 

### Recognizing Mentions of Named Entities in Text through Spacy

After reading in the WARC files and extracting the texts, the named entities in the text are extracted with the off-the-shelf Named Entity Recognition component of Spacy pipeline. We opted for Spacy's small English pipeline trained on Web texts. It is a transformer-based pipeline that uses pre-trained RoBERTa base. Spacy's Named Entitiy Recognizer is readily available and one of the state-of-art modules for linguistic processing. - ADD INFO ON WHICH CLASSES WE INCLUDED

### Generating Candidate Entities through ElasticSearch

In order to properly generate the links that are related to each of the entities, it is crucial to make a distinction between ambiguous entities and non-ambiguous entities. This implies that popular entities, such as "Washington", will be difficult to find within the Wikidata database, since there are hundreds or thousands of links related to this word alone. "Washington" could refer to the state, the city or the American president. Therefore, for such entities, some disambiguation needs to be performed. For this purpose, the package NLTK will be used and Wordnet's synonym sets will be checked to disambiguate the entities and generate the correct URI's from the wikidata database. However, there will be many words that will not be included in the Wordnet synsets, such as IBM. However, generally speaking, these entities will be so specific that it will be relatively easy to find the correct entity URI's through ElasticSearch. Therefore, the assumption is made that, if an entity is not found within Wordnet, that the correct entities will be found within the first 20 results from ElasticSearch. This assumption was created based on the logic that, if a word is not found within Wordnet, it is not ambiguous enough that it won't simply be found through ElasticSearch and that thus the related URI's can always be found through these means. This was checked manually with various entities, such as "IBM" and "Roger Federer", and the correct entity was found generally within the first 5 "hits". Since this cannot be guaranteed, a safety margin of 15 is taken and simply the first 20 hits will be returned.

The problem with Wordnet's synsets is that, for the most popular entities, there will be a large number of synonym sets. To reduce the amount of synsets that are checked through this disambiguation process, the context of the entity from the HTML page is checked with that of the definitions of the synsets. This check is performed with a methodology similar to the Simplified Lex Algorithm. The program deviates from this algorithm a bit by taking the lemma of all words and removing all words that are not nouns. The lemma is taken to increase the chance of hits being generated by the algorithm. Additionally, it is assumed that, for instance, articles, punctuation or stopwords do not contribute to the meaning or context of the entities and synonyms and thus are excluded. After counting the number of similar words between the context and multiple definitions, only the best 3 synsets, with the highest similarity counts, are returned. If no context is given, only the first 3 synsets will be considered (which are the most popular ones according to Wordnet).

However, another problem is that there may not be enough synonyms to choose between. It could be that there is at most 1 match, which is only corresponds to the entity itself. For instance, "Glasgow" has 1 synset, which is the Glasgow synset itself. If this is the case, then no extra synonyms can be retrieved via synsets. To still obtain the correct links, the definition entry will be checked for nouns. Logically speaking, the definition includes nouns that relate Glasgow and thus these words can be used, in conjunction with the entity itself, to hopefully generate the most relevant or correct entity links. The nouns are extracted by using NLTK tokenization and removing all the stopwords from the sentence using "stopwords" corpora for English. This methodology was tested for various entities, such as Glasgow, with great success, as it generally finds the correct entity within at least the first 3 hits for a specific combination. Since again no guarantees can be made, the number of hits should be at least 10. If there are enough synonyms, such as for the entity "dog", the nouns in the definitions may not be used at all, as this would significantly slow down the process. Therefore, only the first 13 unique nouns will be used, in combination with the original entity, to generate the most relevant URI's. Again, this was tested rigorously through trial and error and these numbers showed the best performance in the end. 

### Model tests and choices

Given contextualized embeddings are currently the most sophisticated representations for textual input, we decided to experiment with using contextualized embeddings to represent both the entity mention to be disambiguate and its respective candidate entities. (...)
