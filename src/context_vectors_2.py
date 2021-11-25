import pickle
import torch
import time
from transformers import BertTokenizer, BertModel, BertConfig
from scipy.spatial import distance

def get_text_embedding(text, from_n_layer_on=2):

    """
    Given a text, create its BERT embedding.
    The embedding is the result of averaging the hidden states over the 2nd (default)
    to last BERT layers, followed by averaging over the tokens to get a single vector representation.

    :param str text: string to create an embedding representation
    :param int from_n_layer_on: from which layer of BERT on should the representations be averaged over.
    :return: torch tensor of 768 dimensions
    """

    text_tokens = tokenizer.encode_plus(text,
                                        add_special_tokens=True,
                                        padding=False,
                                        truncation=True,
                                        return_tensors='pt')
    text_tensor = text_tokens['input_ids']

    with torch.no_grad():
        outputs = model(text_tensor) # input tensor with token ids to BERT model
        hidden_states = outputs.hidden_states # get output hidden states
        token_embeddings = torch.stack(hidden_states[from_n_layer_on:], dim=0)  # concatenate tensors of wanted layers
        # print(token_embeddings.size())
        token_embeddings = torch.squeeze(token_embeddings, dim=1)  # get rid of batch dimension
        # print(token_embeddings.size())
        token_embeddings = torch.mean(token_embeddings, dim=0)  # average over layers
        # print(token_embeddings.size())
        text_vector = torch.mean(token_embeddings, dim=0)  # average over all tokens -> tensor size [768]

    return text_vector

def get_representation(span, text):

    """
    Given a span and a text, create a BERT embedding of both.

    In the case the embedding represents the entity mention we want to disambiguate, the span corresponds to
    the mention span identified with a NER system, and the text corresponds the sentential context it is in.

    In the case the embedding represents the entity candidate, the span corresponds to the entity name, and the text
    corresponds the entity description.
    Entity name corresponds to the field "schema_name" in ElasticSearch, and entity description corresponds
    to the field "schema_description" in ElasticSearch.

    The returned embedding is a vector of 1596 dimensions, resulting from concatenating two vectors of 768 dimensions.
    One 768 dimensions vector represents the mention span or the entity name. The other 768 dimensions vector represents
    the context the mention is in or the entity description.

    :param str span: mention span or entity name
    :param str text: mention sentential context or entity description
    :return: torch tensor of 1596 dimensions
    """

    span_vector = get_text_embedding(span,2) # 768 dimensions representing mention span or entity name
    text_vector = get_text_embedding(text,2) # 768 dimensions representing mention sentential context or entity description
    representation = torch.cat((span_vector,text_vector)) # concatenate vectors -> 1596 dimensions

    return representation

def get_similarity_scores(mention,context,candidates):

    # initialize BERT
    MODEL_NAME = 'bert-base-uncased'
    config = BertConfig.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model = BertModel.from_pretrained(MODEL_NAME, config=config)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model.eval()

    mention_vec = get_representation(mention, context)

    for mention_info in text[1]['entities']:
        mention = mention_info[0]
        context = mention_info[2]
        mention_vec = get_representation(mention,context)
        # generate candidate vectors and measure pairwise similarity between mention and candidate vectors
        similarity_scores = []
        print(f'---------------------------------')
        print(f'Mention: {mention}')
        print()
        for dict in entity_list[counter]:
            candidate_vec = get_representation(dict['name'],dict['description'])
            similarity = 1 - distance.cosine(mention_vec,candidate_vec)
            similarity_scores.append(similarity)
            print(f'Entity: {dict["name"]}  {similarity}')
        counter += 1
