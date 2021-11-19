# README Group 3 Assignment 1

In this readme, all of the installation instructions and coding choices will be further elaborated. 


## Installation

TODO 

## Coding Choices


### Importing and reading WARC files

TODO

### Generating Candidate Entities through ElasticSearch

In order to properly generate the links that are related to each of the entities, it is crucial to make a distinction between ambiguous entities and non-ambiguous entities. This implies that popular entities, such as "Washington" will be difficult to find within the Wikidata database, since there are hundreds of links related to this word alone. "Washington" could refer to the state, the city or the American president. Therefore, for such entities, some disambiguation needs to be performed. For this purpose, the package NLTK will be used and thus Wordnet synsets will be checked to generate corresponding links. However, there will be many words that will not be included in the Wordnet synsets, such as IBM. However, generally speaking, these entities will be so specific that it will be relatively easy finding the correct entity URI's through ElasticSearch. Therefore, the assumption is made that, if an entity is not found within Wordnet, that the correct entities will be found within the first 20 results from ElasticSearch. This conclusion was drawn based on the logic that, if a word is not found within Wordnet, it is not ambiguous enough that it won't simply be found through ElasticSearch and that thus the related URI's can always be found through these means. This was checked manually with various entities, such as "IBM" and "Roger Federer", and the correct entity was found generally within the first 5 "hits".  Since this cannot be guaranteed, a safety margin of 15 is taken and simply the first 20 hits will be returned. 

However, if there is at least 1 match within Wordnet for the current entity, a few choices had to be made. First, if there is at least 1 match, the word itself is also included within these synsets. For instance, "Glasgow" has 1 synset, which is the Glasgow synset itself. If this is the case, then no synonyms can be retrieved via synsets. To still obtain the correct links, the definition entry will be checked for nouns. Logically speaking, the definition includes nouns that relate Glasgow and thus these words can be used, in conjunction with the entity itself, to hopefully generate the correct entity links.  The nouns are extracted by using  NLTK tokenization and removing all the stopwords from the sentence using "stopwords" corpora for English. This methodology was tested for various entities, such as Glasgow, with great success, as it generally finds the correct entity within at least the first 3 hits for a specific combination. Since again no guarantees can be made, the number of hits should be at least 10. If there are enough synonyms, such as for the entity "dog", the nouns in the definitions will not be used, as this would significantly slow down the process. 

### Model tests and choices
