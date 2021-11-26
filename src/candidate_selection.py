import torch
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from nltk.tokenize import word_tokenize
import numpy as np
from scipy.spatial import distance
import requests
import os.path
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from .entity_generation_ES import order_list_from_list

stop_words = set(stopwords.words("english"))
stop_words.add("-")
lemmatizer = WordNetLemmatizer()
punctuation = ['!', '/', '%', '|', '\\', ']', '[', '^', '<', '{', '}', '~', '`', '(', ')',
               '"', '=', '>', ';', '@', '\'', '*', '+', '?', '_', '...', ',', '--', ':']

def get_bert_embedding(text, from_n_layer_on, model, tokenizer):

    """
    Given a text, create its BERT embedding.
    The embedding is the result of averaging the hidden states over the 2nd (default)
    to last BERT layers, followed by averaging over the tokens to get a single vector representation.

    :param str text: string to create an embedding representation
    :param int from_n_layer_on: from which layer of BERT on should the representations be averaged over.
    :param model: (BERT) language model to get embeddings from
    :param tokenizer: (BERT) tokenizer to tokenize text strings and input tokens to model

    :return: torch tensor of 768 dimensions
    """

    text_tokens = tokenizer.encode_plus(text,
                                        add_special_tokens=True,
                                        padding=False,
                                        truncation=True,
                                        return_tensors='pt')
    text_tensor = text_tokens['input_ids']

    with torch.no_grad():
        # input tensor with token ids to BERT model
        outputs = model(text_tensor)
        # get output hidden states
        hidden_states = outputs.hidden_states
        # concatenate tensors of wanted layers
        token_embeddings = torch.stack(hidden_states[from_n_layer_on:], dim=0)
        # get rid of batch dimension
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        # average over layers
        token_embeddings = torch.mean(token_embeddings, dim=0)
        # average over all tokens -> tensor size [768]
        text_vector = torch.mean(token_embeddings, dim=0)

    return text_vector

def get_bert_representation(span, text, model, tokenizer, from_layer_n_on = 2):

    """
    Given a span and a text, create a BERT embedding of both.

    In the case the embedding represents the entity mention we want to disambiguate, the span corresponds to
    the mention span identified with a NER system, and the text corresponds the sentential context it is in.

    In the case the embedding represents the entity candidate, the span corresponds to the entity name, and the text
    corresponds the entity description.
    Entity name corresponds to the field "schema_name" in ElasticSearch, and entity description corresponds
    to the field "schema_description" in ElasticSearch.

    The returned embedding is a vector of 1596 dimensions, resulting from concatenating two vectors of 768 dimensions.
    One 768-dimensions vector represents the mention span or the entity name. The other 768-dimensions vector represents
    the context the mention is in or the entity description.

    :param str span: mention span or entity name
    :param str text: mention sentential context or entity description
    :param model: (BERT) language model to get embeddings from
    :param tokenizer: (BERT) tokenizer to tokenize text strings and input tokens to model
    :param int from_n_layer_on: from which layer of BERT on should the representations be averaged over.

    :return: torch tensor of 1596 dimensions
    """

    # 768 dimensions representing mention span or entity name
    span_vector = get_bert_embedding(span, from_layer_n_on, model, tokenizer)
    # 768 dimensions representing mention sentential context or entity description
    text_vector = get_bert_embedding(text, from_layer_n_on, model, tokenizer)
    # concatenate vectors -> 1596 dimensions
    representation = torch.cat((span_vector,text_vector))

    return representation

def get_glove_embedding(text,glove_model):

    """
    Given a text, create its glove embedding. The embedding is the result of averaging the representations
    over the tokens in text to get a single vector representation.

    :param str text: string to create an embedding representation
    :param model: word embedding model to get embeddings from

    :return: numpy array of 100 dimensions
    """

    # tokenize text with nltk tokenizer
    tokens_span = word_tokenize(text.lower())
    token_vectors = []
    # for each token get glove embedding and append to list
    for token in tokens_span:
        if token in glove_model.keys():
            token_vec = glove_model[token]
            token_vectors.append(token_vec)
    # if token out of the model's vocabulary, represent it with a numpy array of 0's
    if len(token_vectors) == 0:
        text_vector = np.zeros(100)
    else:
        # cast list of token embeddings into a numpy array
        token_vecs = np.asarray(token_vectors)
        # average over tokens -> numpy array of 100 dimensions
        text_vector = np.mean(token_vecs, axis=0)

    return text_vector

def get_glove_representation(span,text,glove_model):

    """
    Given a span and a text, create a glove embedding of both.

    In the case the embedding represents the entity mention we want to disambiguate, the span corresponds to
    the mention span identified with a NER system, and the text corresponds the sentential context it is in.

    In the case the embedding represents the entity candidate, the span corresponds to the entity name, and the text
    corresponds the entity description.
    Entity name corresponds to the field "schema_name" in ElasticSearch, and entity description corresponds
    to the field "schema_description" in ElasticSearch.

    The returned embedding is a vector of 200 dimensions, resulting from concatenating two vectors of 100 dimensions.
    One 100-dimensions vector represents the mention span or the entity name. The other 100-dimensions vector represents
    the context the mention is in or the entity description.

    :param str span: mention span or entity name
    :param str text: mention sentential context or entity description
    :param glove_model: loaded glove model of 100-dimensions vectors

    :return: numpy array of 200 dimensions
    """

    span_vec = get_glove_embedding(span,glove_model)
    text_vec = get_glove_embedding(text,glove_model)
    representation = np.append(span_vec, text_vec)

    return representation

def extract_nouns_schemas(list_of_schemas):

    is_noun = lambda pos:pos[:2] == "NN"
    return [ [word for (word,pos) in nltk.pos_tag(nltk.word_tokenize(schema)) \
            if word.strip() not in stop_words and word.strip() not in punctuation and not word.strip().isdigit() and is_noun ] \
            for schema in list_of_schemas]

def find_best_match(clean_context, list_of_schemas, list_of_dicts):

    list_of_counts = []

    for schema in list_of_schemas:
        similarity_count = sum( [1 if lemmatizer.lemmatize(word) in clean_context else 0 for word in schema])
        list_of_counts.append(similarity_count)

    return order_list_from_list(list_of_dicts, list_of_counts, True)[0]['uri']


def get_best_candidate(mention, context, candidates, method, model, tokenizer):

    """
    Given a named entity mention and its entities candidates, return the best entity candidate, according to a method
    defined in parse argument.

    :param str mention: mention to be disambiguated.
    :param str context: sentence mention is in.
    :param list of dicts candidates: each candidate is represented as a dict with keys: uri, name, rdfs, description
    :param method: which method to perform candidate selection with
    :param model: the loaded model. If method is 'lesk' or 'popularity', model is None.
    :param tokenizer: the tokenizer object. If method is 'lesk', 'popularity' or 'glove', tokenizer is None.

    :return: the wikidata uri of candidate entity with highest similarity score to the mention
    """

    best_candidate = None

    if method == 'popularity':
       # candidates in dict are ordered by popularity, thus the first candidate is the best according to popularity
        best_candidate = candidates[0]['uri']

    elif method == 'lesk':

        schema_list = [x["description"] for x in candidates]
        clean_schema_list = extract_nouns_schemas(schema_list)
        clean_context = extract_nouns_schemas([context])
        best_candidate = find_best_match(clean_context, clean_schema_list, candidates)

    elif method == 'bert':

        mention_vector = get_bert_representation(mention, context, model, tokenizer)

        max_similarity = 0

        for candidate_dict in candidates:
            if 'name' in candidate_dict.keys():
                name = candidate_dict['name']
            else:
                name = candidate_dict['rdfs']
            description = candidate_dict['description']
            candidate_vector = get_bert_representation(name, description, model, tokenizer)

            similarity = 1 - distance.cosine(mention_vector, candidate_vector)
            if similarity > max_similarity:
                max_similarity = similarity
                best_candidate = candidate_dict['uri']

    elif method == 'glove':

        mention_vector = get_glove_representation(mention, context, model)

        max_similarity = 0

        for candidate_dict in candidates:
            if 'name' in candidate_dict.keys():
                name = candidate_dict['name']
            else:
                name = candidate_dict['rdfs']
            description = candidate_dict['description']
            candidate_vector = get_glove_representation(name, description, model)

            similarity = 1 - distance.cosine(mention_vector, candidate_vector)
            if similarity > max_similarity:
                max_similarity = similarity
                best_candidate = candidate_dict['uri']

    return best_candidate


def candidate_selection(warc_texts, candidate_dict, method):

    """
    Given text files and dictionary of entity candidates, call get_best_candidate for each mention in each text
    with the method specified on method parameter. Tries to rank the URI's by disambiguating the entities obtained through ElasticSearch
    If method is a language model, load the language model for further processing.

    :param warc_texts: a list of dictionaries containing the webpage-id's and corresponding entities
    :param candidate_dict: a dictionary with all results of the ElasticSearch
    :param method: The method to be used, the default is 'popularity'

    :return: returns a list of triples, containing the web page id, the entity and the wikidata URI
    """

    if method == 'bert':
        print("Loading DistilBERT Model...")
        MODEL_NAME = 'distilbert-base-uncased'
        config = DistilBertConfig.from_pretrained(MODEL_NAME, output_hidden_states=True)
        model = DistilBertModel.from_pretrained(MODEL_NAME, config=config)
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        model.eval()
        print("DistilBERT loaded!")

    if method == 'glove':
        MODEL_FILE = 'assets/glove.6B.100d.txt'
        if not os.path.isfile(MODEL_FILE):
            r = requests.get('http://nlp.stanford.edu/data/glove.6B.zip', allow_redirects=True)
            open(MODEL_FILE, 'wb').write(r.content)
        print("Loading Glove Model...")
        model = {}
        with open(MODEL_FILE, 'r') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                model[word] = embedding
        print(f"{len(model)} words loaded!")
        tokenizer = None

    else: # if method == 'popularity' or 'lesk'
        model = None
        tokenizer = None


    output = []
    for text_id, entity_list in warc_texts.items():
        for entity_tuple in entity_list:

            mention, label, context = entity_tuple

            if mention not in candidate_dict.keys():
                continue

            candidates = candidate_dict[mention]
            if not candidates:
                continue

            best_candidate = get_best_candidate(mention, context, candidates, method, model, tokenizer)
            if not None:
                output.append((text_id, mention, best_candidate))

    return output