# Assignment 1 Entity Linking ; Group 3

This repository contains the solution of Group 3 for the Entity Linking Assignment from the course Web Processing Data Systems 2021 at VU Amsterdam. Our solution recognizes named entity mentions in Web pages and link them to Wikidata. In the following sections, the installation instructions and coding choices will be further elaborated. Note that this was elaborately tested on Windows specifically, since neither group member owns a Macbook or has a separate linux installation. Therefore, it was not possible to check whether it properly runs on other Operating Systems.  

## Structure directory

To understand how the directory is structured, a small summary will be provided here. The _assets_ folder contains all the wikidata information and the optional local elasticsearch mechanisms, which have not been altered. The _data_ directory contains all the warc files, including the sample warc file. This has not been altered either. The _old files_ directory contains all the original files, such as the _starter_code_, _test_sparql_, etc. of the original directory, which are currently not in use by our program. The _outputs_ directory provides all the pickle files that have been saved. These files include the processed WARC files and the ElasticSearch candidates for each entity in the processed WARC files. The _src_ directory contains all of the major code snippets used at some point throughout the project. The files in the main directory includes files that are used by the starter_code.py file, which executes the main program. 

## Installation packages and data

In order to simplify the process of install everything required for this assignment, a shell script has been created. If this is your first time running this script, simply execute in your docker image the following code:
`sh run_entity_link_script.sh`

This will execute the following code:
- Installs all packages contained in the requirements.txt file
- Downloads the Spacy model for the Named entity recognition process
- Downloads the dataset and information for the various NLTK functions
- Executes the starter_code.py script

These processes can also be executed manually if something is not going according to plan. Please note that, for some packages, there needs to be some installation for Microsoft Visual C++ build > 14.0. If this is not yet installed on your laptop, do this first or you will get an error, at least on Windows, for not having it installed.

## Code Structure

Our code starts by parsing the arguments of the commandline/terminal. There are 4 arguments. For more information, check the `-h` or `--help` command for more information or consult the code. The idea is to first process the warc files found within the data/warcs folder. Ideally, there are multiple warc files so that the process is sped up, since this code can be ran in parallel. This is done within the _extraction.py_ file in the _outputs_ directory. Afterwards, the entity generation is performed from _entity\_generation\_ES.py, which generates various candidate entities through multiple elastic searches. Lastly, the candidates will be ranked according to the model proposed by the user through the command line. The fastest models are "popularity" and "lesk" and are the ones we propose to prevent excessive load times.

## Coding Choices

In this section, we explain how we approach the challenges of candidate generation and disambiguation and motivate our choices.

### Importing and reading WARC files

TODO

### Recognizing Mentions of Named Entities in Text through Spacy

After reading in the WARC files and extracting the texts, the named entities in the text are extracted with the off-the-shelf Named Entity Recognition component of Spacy pipeline. We opted for Spacy's small English pipeline trained on Web texts. It is a transformer-based pipeline that uses pre-trained RoBERTa base. Spacy's Named Entitiy Recognizer is readily available and one of the state-of-art modules for linguistic processing. - ADD INFO ON WHICH CLASSES WE INCLUDED

### Generating Candidate Entities through ElasticSearch

In order to properly generate the links that are related to each of the entities, it is crucial to make a distinction between ambiguous entities and non-ambiguous entities. This implies that popular entities, such as "Washington", will be difficult to find within the Wikidata database, since there are hundreds or thousands of links related to this word alone. "Washington" could refer to the state, the city or the American president. Therefore, for such entities, some disambiguation needs to be performed. For this purpose, the package NLTK will be used and Wordnet's synonym sets will be checked to disambiguate the entities and generate the correct URI's from the wikidata database. However, there will be many words that will not be included in the Wordnet synsets, such as IBM. However, generally speaking, these entities will be so specific that it will be relatively easy to find the correct entity URI's through ElasticSearch. Therefore, the assumption is made that, if an entity is not found within Wordnet, that the correct entities will be found within the first 20 results from ElasticSearch. This assumption was created based on the logic that, if a word is not found within Wordnet, it is not ambiguous enough that it won't simply be found through ElasticSearch and that thus the related URI's can always be found through these means. This was checked manually with various entities, such as "IBM" and "Roger Federer", and the correct entity was found generally within the first 5 "hits". Since this cannot be guaranteed, a safety margin of 15 is taken and simply the first 20 hits will be returned.

The problem with Wordnet's synsets is that, for the most popular entities, there will be a large number of synonym sets. To reduce the amount of synsets that are checked through this disambiguation process, the context of the entity from the HTML page is checked with that of the definitions of the synsets. This check is performed with a methodology similar to the Simplified Lex Algorithm. The program deviates from this algorithm a bit by taking the lemma of all words and removing all words that are not nouns. The lemma is taken to increase the chance of hits being generated by the algorithm. Additionally, it is assumed that, for instance, articles, punctuation or stopwords do not contribute to the meaning or context of the entities and synonyms and thus are excluded. After counting the number of similar words between the context and multiple definitions, only the best 3 synsets, with the highest similarity counts, are returned. If no context is given, only the first 3 synsets will be considered (which are the most popular ones according to Wordnet).

However, another problem is that there may not be enough synonyms to choose between. It could be that there is at most 1 match, which is only corresponds to the entity itself. For instance, "Glasgow" has 1 synset, which is the Glasgow synset itself. If this is the case, then no extra synonyms can be retrieved via synsets. To still obtain the correct links, the definition entry will be checked for nouns. Logically speaking, the definition includes nouns that relate Glasgow and thus these words can be used, in conjunction with the entity itself, to hopefully generate the most relevant or correct entity links. The nouns are extracted by using NLTK tokenization and removing all the stopwords from the sentence using "stopwords" corpora for English. This methodology was tested for various entities, such as Glasgow, with great success, as it generally finds the correct entity within at least the first 3 hits for a specific combination. Since again no guarantees can be made, the number of hits should be at least 8. If there are enough synonyms, such as for the entity "dog", the nouns in the definitions may not be used at all, as this would significantly slow down the process. Therefore, only the first 8 unique nouns will be used, in combination with the original entity, to generate the most relevant URI's. Again, this was tested rigorously through trial and error and these numbers showed the best performance in the end. These numbers were not increased, since this would result in large time costs per entity.

### Model tests and choices

#### Unsupervised Methods

Given a recognized mention and its respective candidate entities, a candidate entity needs to be selected to disambiguate the mention. This selection is done using unsupervised methods, since we are dealing with a large amount of unannotated data on which a supervised model is unlikely to generalize well. In particular, we opted to experiment with a **Vector Space Model (VSM)** approach, given its execution simplicity. The idea is that both mention and candidate entities are represented with vectors in a shared space. Then, we calculate the **similarity** between the mention vector and each candidate vector and select the candidate entity with the highest similarity score. 

#### Creating Vectors

To generate such vectors, we experimented with encoding the mention and candidate entities with a language model. To represent the mention, the **mention span** and the **sentence** which it was in were taken. To represent the entity, the **entity name** and its **description** were taken. The entity name and description were obtained from the fields "schema_name" (or "rdsf_label" in case there was no "schema_name") and "schema_description" on ElasticSearch. An embedding representation was generated of the tokens of each of these elements (mention span, mention sentential context, entity name, and entity description), and averaged over each token embedding to get a fixed-length vector representation of each of the elements. To represent the mention, the resulting averaged vector of the mention span and the resulting averaged vector of its sentence were concatenated.  To represent the entity, the same was done with the averaged vectors of the entity name and description.

#### Context-dependent Features

We experimented with two language models to generate such embeddings. The first was BERT. BERT is a state-of-the-art Transformer-based language model that generates contextualized embeddings. This means that each embedding it generates given a string contains information on the left and right textual context of that string. However, contextualized embeddings can be very expensive to compute if dealing with large amounts of data, which is the case here. For that reason, we switched to a ligher version of BERT, named **DistilBERT**, which is around 40% faster while attaining 95% of the accuracy. Even with this ligher version, it took about 30 minutes to encode only the mentions of one of the warc files (with an 8GB RAM). Because of the expensiveness to calculate contextualized embeddings, we also experimented with static embeddings, namely **GloVe**. Static embeddings are also representations of words in a dense embedded space, but, once trained, they do not adjust to unseen input. We used GloVe with 100 dimensional vectors pre-trained on Wikipedia (2014) and Gigaword 5. They turned out to be quite expensive to compute, taking about 18 minutes to encode the mentions of one of the warc files.

#### WordNet Popularity

Given these scalability issues, we decided to implement simpler but less computationally expensive approaches to candidate selection.  The first is based on WordNet popularity. Since we use WordNet in our candidate generation module, the candidate entities are ordered according to its frequency on the corpus WordNet uses. Therefore, by simply selecting the first key, we obtain the best candidate entity according to WordNet popularity. However, for those mentions not present in WordNet, the candidates are ordered according to the relevance scores given by ElasticSearch, which do not only take popularity into account.

#### Simplified Lesk Algorithm

Another less expensive approach we implement is a method based on the Simplified Lesk Algorithm. It simply counts how many tokens in the sentential context of the mention are also in the entity description. The entity candidate whose description has the highest overlap with the mention context is the selected candidate. More specifications of this method was already mentioned in the section Generating Candidates through ElasticSearch. 

## Results


