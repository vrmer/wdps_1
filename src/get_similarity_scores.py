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

# def get_mention_vector(mention,context):
#
#     """
#     Given a mention span and the sentential context it is in, create a BERT embedding to represent the mention.
#     This embedding is a vector of 1596 dimensions, resulting from concatenating a vector of 768 dimensions that
#     represents the mention span, with a vector of 768 dimensions that represents the context the mention is in.
#     Each mention or context vector is the result of averaging the hidden states over the 2nd to last BERT layers,
#     followed by averaging over the tokens to get a single vector representation.
#
#     :param str mention: mention span
#     :param str context: sentence mention span is in
#     :return: torch tensor of 1596 dimensions
#     """
#
#     # tokenized_text = [tokenizer.cls_token] + tokenizer.tokenize(context) + [tokenizer.sep_token]
#     # tokenized_span = [tokenizer.cls_token] + tokenizer.tokenize(mention) + [tokenizer.sep_token]
#     # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
#     # indexed_span = tokenizer.convert_tokens_to_ids(tokenized_span)
#     # tokens_tensor = torch.tensor(indexed_tokens).unsqueeze(0)
#     # print(tokens_tensor.size())
#     # span_tensor = torch.tensor(indexed_span).unsqueeze(0)
#     # print(span_tensor.size())
#
#     context_tokens = tokenizer.encode_plus(context,
#                                            add_special_tokens = True,
#                                            padding = False,
#                                            truncation = True,
#                                            return_tensors = 'pt')
#     span_tokens = tokenizer.encode_plus(mention,
#                                         add_special_tokens = True,
#                                         padding = False,
#                                         truncation = True,
#                                         return_tensors = 'pt')
#
#     tokens_tensor = context_tokens['input_ids']
#     span_tensor = span_tokens['input_ids']
#
#     # # mention = mention.split()
#     # # print('-----------------------------')
#     # # print(mention)
#     # # print(context)
#     # # print()
#     # mention = tokenizer.tokenize(mention)
#     # # print(mention)
#     # # print(mention)
#     # mention_idx = []
#     # for subword in mention:
#     #     # print(word.lower())
#     #     if subword.lower() in tokenized_text:
#     #         mention_idx.append(tokenized_text.index(subword.lower()))
#     #         # print(mention_idx)
#     #     else: # if word out of model vocab, take the index of the last word piece
#     #         for token in enumerate(tokenized_text):
#     #             # print(token)
#     #             if token[0:2] == '##' and tokenized_text[token[0] + 1][0:2] != '##':
#     #                 mention_idx.append(tokenized_text.index(token))
#     #                 # print(mention_idx) # [768 -> mention span, 768 -> context]
#
#     # print(mention_idx)
#
#     # with torch.no_grad():
#     #     outputs = model(tokens_tensor)
#     #     hidden_states = outputs.hidden_states
#     #     context_vector = hidden_states[12][0]
#     #     # print(len(context_vector))
#     #     span_vecs = context_vector[mention_idx[0]:mention_idx[-1]+1]
#     #     # print(len(span_vecs))
#     #     span_vec = torch.mean(span_vecs,0)
#     #     # print(span_vec.size())
#     #     cls_vec = context_vector[0]
#     #     # print(cls_vec.size())
#     #     mention_vec = torch.cat((span_vec,cls_vec)) # concatenate span and cls vectors to represent mention
#     #     # print(mention_vec.size())
#
#     token_embeddings = pooling_token_embeddings(tokens_tensor,2)
#     context_vector = torch.mean(token_embeddings, dim=0) # average over all tokens [768]
#     # span_vector = token_embeddings[mention_idx[0]:mention_idx[-1]+1] # select tokens in mention span
#     token_embeddings = pooling_token_embeddings(span_tensor,2)
#     span_vector = torch.mean(token_embeddings,dim=0) # average over the span tokens [768]
#     # print(context_vector.size())
#     # print(span_vector.size())
#     mention_vec = torch.cat((span_vector,context_vector)) # concatenate span and context vectors to represent mention [15..]
#
#     return mention_vec

# def get_candidate_vector(name,description):
#
#     """
#     Given an entity name and its description, create a BERT embedding to represent the entity. The entity name
#     corresponds to the field "schema_name" in ElasticSearch, and the description corresponds to the field "schema
#     _description" in ElasticSearch. The embedding is a vector of 1596 dimensions, resulting from concatenating a
#     vector of 768 dimensions that represents the entity name, with a vector of 768 dimensions that represents the
#     entity description. Each mention or context vector is the result of averaging the hidden states over the 2nd
#     to last BERT layers, followed by averaging over the tokens to get a single vector representation.
#
#     :param str name: entity name
#     :param str description: entity description
#     :return: torch tensor of 1596 dimensions
#     """
#
#     # name_tokens = [tokenizer.cls_token] + tokenizer.tokenize(name) + [tokenizer.sep_token]
#     # desc_tokens = [tokenizer.cls_token] + tokenizer.tokenize(description) + [tokenizer.sep_token]
#     # # print(name_tokens, desc_tokens)
#     # name_ids = tokenizer.convert_tokens_to_ids(name_tokens)
#     # desc_ids = tokenizer.convert_tokens_to_ids(desc_tokens)
#     # name_tensor = torch.tensor(name_ids).unsqueeze(0)
#     # desc_tensor = torch.tensor(desc_ids).unsqueeze(0)
#
#     desc_tokens = tokenizer.encode_plus(description,
#                                            add_special_tokens=True,
#                                            padding=False,
#                                            truncation=True,
#                                            return_tensors='pt')
#     name_tokens = tokenizer.encode_plus(name,
#                                         add_special_tokens=True,
#                                         padding=False,
#                                         truncation=True,
#                                         return_tensors='pt')
#
#     name_tensor = name_tokens['input_ids']
#     desc_tensor = desc_tokens['input_ids']
#
#     # with torch.no_grad():
#     #     outputs = model(name_tensor)
#     #     name_vec = outputs.hidden_states[12][0][0] # CLS token
#     #     # print(len(name_vec))
#     #     outputs = model(desc_tensor)
#     #     desc_vec = outputs.hidden_states[12][0][0]
#     #     # print(desc_vec.size())
#     #     candidate_vec = torch.cat((name_vec,desc_vec)) # concatenate name and description vectors to represent mention
#     #     # print(candidate_vec.size())
#
#
#     token_embeddings = pooling_token_embeddings(name_tensor, 2)
#     name_vector = torch.mean(token_embeddings, dim=0)  # average over all tokens in name
#     token_embeddings = pooling_token_embeddings(desc_tensor,2)
#     desc_vector = torch.mean(token_embeddings, dim=0) # average over all tokens in description
#     candidate_vec = torch.cat((name_vector, desc_vector))  # concatenate name and description vectors to represent mention
#
#     return candidate_vec

if __name__ == '__main__':

    PKL_FILE = 'outputs/CC-MAIN-20201001210429-20201002000429-00799_entities.pkl'

    # initialize BERT
    MODEL_NAME = 'bert-base-uncased'
    config = BertConfig.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model = BertModel.from_pretrained(MODEL_NAME, config=config)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model.eval()

    # generate mention vectors
    with open(PKL_FILE, 'rb') as infile:
        texts = pickle.load(infile)

    start = time.perf_counter()

    # counter = 0
    # c = 0
    for key, entities in texts.items():
        for lb, entity_list in entities.items():
            for entity_tuple in entity_list:
                try:
                    mention, label, context = entity_tuple
                    # c += 1
                    # print(c)
                except ValueError:
                    # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    # print(entity_list)
                    # print(type(entity_list))
                    # # counter += 1
                    # # print(counter)
                    exit()
                try:
                    mention_vec = get_representation(mention, context)
                except RuntimeError:
                    print(mention)
                    print(context)
                    print('------------------------------------------')
                    print()

    end = time.perf_counter()

    total_time = end - start
    print(f'Total time spent in encoding: {total_time}')


    # # generate mention vectors
    # # texts = read_pkl_files(PKL_FILE)
    # text = ('uri', {'entities':[('The Washington Post','ORG','The Washington Post: Democracy Dies in Darkness'),
    #                             ('UK','ORG','Covid-19 vaccines are licensed in the UK only for children aged 12'),
    #                             ('Big Blue', 'ORG', 'Since then, Big Blue path the way for technological revolution.')]})
    #
    # entity_list = [[{'name': 'Washington', 'description': 'state of the United States of America'},
    #                {'name': 'George Washington', 'description': '1st president of the United States (1732−1799)'},
    #                {'name': 'The Washington Post', 'description': 'daily broadsheet newspaper in Washington, D.C.'}],
    #                [{'name': 'United Kingdom', 'description': 'country in Western Europe'},
    #                 {'name': 'UK Independence Party', 'description': 'British political party'},
    #                 {'name': 'Ukrainian', 'description': 'East Slavic language'}],
    #                [{'name': 'IBM', 'description': 'American multinational technology and consulting corporation'},
    #                 {'name': 'The Big Blue', 'description': '1988 English-language film directed by Luc Besson'},
    #                 {'name': 'Big Blue', 'description': 'painting by Ronald Davis'}]]
    # counter = 0
    # for mention_info in text[1]['entities']:
    #     mention = mention_info[0]
    #     context = mention_info[2]
    #     mention_vec = get_representation(mention,context)
    #     # generate candidate vectors and measure pairwise similarity between mention and candidate vectors
    #     similarity_scores = []
    #     print(f'---------------------------------')
    #     print(f'Mention: {mention}')
    #     print()
    #     for dict in entity_list[counter]:
    #         candidate_vec = get_representation(dict['name'],dict['description'])
    #         similarity = 1 - distance.cosine(mention_vec,candidate_vec)
    #         similarity_scores.append(similarity)
    #         print(f'Entity: {dict["name"]}  {similarity}')
    #     counter += 1
