import pickle
import torch
import time
from transformers import BertTokenizer, BertModel, BertConfig
from scipy.spatial import distance


def read_pkl_files(pickle_file):

    with open(pickle_file, 'rb') as infile:
        texts = pickle.load(infile)
        for text in texts.items():
            print(text)
            exit(3)
    return texts

def get_mention_vector(mention,context):

    tokenized_text = [tokenizer.cls_token] + tokenizer.tokenize(context) + [tokenizer.sep_token]
    # print(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor(indexed_tokens).unsqueeze(0)

    mention = mention.split()
    mention_idx = []
    for word in mention:
        if word.lower() in tokenized_text:
            mention_idx.append(tokenized_text.index(word.lower()))
            # print(mention_idx)
        else: # if word out of model vocab, take the index of the last word piece
            for token in enumerate(tokenized_text):
                if token[0:2] == '##' and tokenized_text[token[0] + 1][0:2] != '##':
                    mention_idx.append(tokenized_text.index(token))
                    # print(mention_idx) # [768 -> mention span, 768 -> context]

    # with torch.no_grad():
    #     outputs = model(tokens_tensor)
    #     hidden_states = outputs.hidden_states
    #     context_vector = hidden_states[12][0]
    #     # print(len(context_vector))
    #     span_vecs = context_vector[mention_idx[0]:mention_idx[-1]+1]
    #     # print(len(span_vecs))
    #     span_vec = torch.mean(span_vecs,0)
    #     # print(span_vec.size())
    #     cls_vec = context_vector[0]
    #     # print(cls_vec.size())
    #     mention_vec = torch.cat((span_vec,cls_vec)) # concatenate span and cls vectors to represent mention
    #     # print(mention_vec.size())

    token_embeddings = pooling_token_embeddings(tokens_tensor,2)
    context_vector = torch.mean(token_embeddings, dim=0) # average over all tokens [768]
    span_vector = token_embeddings[mention_idx[0]:mention_idx[-1]+1] # select tokens in mention span
    span_vector = torch.mean(span_vector,dim=0) # average over the span tokens [768]
    # print(context_vector.size())
    # print(span_vector.size())
    mention_vec = torch.cat((span_vector,context_vector)) # concatenate span and context vectors to represent mention [15..]

    return mention_vec

def get_candidate_vector(name,description):

    name_tokens = [tokenizer.cls_token] + tokenizer.tokenize(name) + [tokenizer.sep_token]
    desc_tokens = [tokenizer.cls_token] + tokenizer.tokenize(description) + [tokenizer.sep_token]
    # print(name_tokens, desc_tokens)
    name_ids = tokenizer.convert_tokens_to_ids(name_tokens)
    desc_ids = tokenizer.convert_tokens_to_ids(desc_tokens)
    name_tensor = torch.tensor(name_ids).unsqueeze(0)
    desc_tensor = torch.tensor(desc_ids).unsqueeze(0)

    # with torch.no_grad():
    #     outputs = model(name_tensor)
    #     name_vec = outputs.hidden_states[12][0][0] # CLS token
    #     # print(len(name_vec))
    #     outputs = model(desc_tensor)
    #     desc_vec = outputs.hidden_states[12][0][0]
    #     # print(desc_vec.size())
    #     candidate_vec = torch.cat((name_vec,desc_vec)) # concatenate name and description vectors to represent mention
    #     # print(candidate_vec.size())

    token_embeddings = pooling_token_embeddings(name_tensor, 2)
    name_vector = torch.mean(token_embeddings, dim=0)  # average over all tokens in name
    token_embeddings = pooling_token_embeddings(desc_tensor,2)
    desc_vector = torch.mean(token_embeddings, dim=0) # average over all tokens in description
    candidate_vec = torch.cat((name_vector, desc_vector))  # concatenate name and description vectors to represent mention

    return candidate_vec

def pooling_token_embeddings (input,from_n_layer_on):

    with torch.no_grad():
        outputs = model(input)
        hidden_states = outputs.hidden_states
        token_embeddings = torch.stack(hidden_states[from_n_layer_on:], dim=0) # concatenate tensors of wanted layers
        # print(token_embeddings.size())
        token_embeddings = torch.squeeze(token_embeddings, dim=1) # get rid of batch dimension
        # print(token_embeddings.size())
        token_embeddings = torch.mean(token_embeddings,dim=0) # average over layers
        # print(token_embeddings.size())

    return token_embeddings


if __name__ == '__main__':

    PKL_FILE = 'outputs/CC-MAIN-20201001210429-20201002000429-00799_entities.pkl'

    # initialize BERT
    MODEL_NAME = 'bert-base-uncased'
    config = BertConfig.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model = BertModel.from_pretrained(MODEL_NAME, config=config)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model.eval()

    # generate mention vectors
    texts = read_pkl_files(PKL_FILE)
    # print(texts)

    start = time.perf_counter()
    for idx, mention_info in enumerate(texts[1]['entities']):
        mention = mention_info[0]
        context = mention_info[2]
        mention_vec = get_mention_vector(mention, context)
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
    #     mention_vec = get_mention_vector(mention,context)
    #     # generate candidate vectors and measure pairwise similarity between mention and candidate vectors
    #     similarity_scores = []
    #     print(f'---------------------------------')
    #     print(f'Mention: {mention}')
    #     print()
    #     for dict in entity_list[counter]:
    #         candidate_vec = get_candidate_vector(dict['name'],dict['description'])
    #         similarity = 1 - distance.cosine(mention_vec,candidate_vec)
    #         similarity_scores.append(similarity)
    #         print(f'Entity: {dict["name"]}  {similarity}')
    #     counter += 1
