import pickle
from collections import defaultdict

# with open('outputs/candidate_dictionary.pkl', 'rb') as f:
#     candidate_dict = pickle.load(f)
# print(candidate_dict['Washington'])
# exit(1)


def merge_pooled_processes(pooled_processes):
    """

    :param pooled_processes:
    :return:
    """
    unique_uris = set()
    merged_processes = defaultdict(list)

    for pooled_process in pooled_processes:
        for target_entity, returned_queries in pooled_process.items():
            for returned_query in returned_queries:
                uri = returned_query['uri']
                if uri not in unique_uris:
                    unique_uris.add(uri)
                    merged_processes[target_entity].append(returned_query)
    return merged_processes


# loading the outcome of pooled processes, a list
with open('outputs/test_dict.pkl', 'rb') as f:
    pooled_processes = pickle.load(f)

# TODO: this is Tim's code that should help filter for uris
# list_of_uris = [dict(t) for t in {tuple(d.items()) for d in list_of_uris}]

to_add = {'uri': '<http://www.wikidata.org/entity/Q61>', 'rdfs': 'Washington, D.C.', 'name': 'Washington, D.C.', 'description': 'capital city of the United States'}

pooled_processes[0]['Washington'].append(to_add)

merged_processes = merge_pooled_processes(pooled_processes)

print(len(merged_processes))

# for key, value in to_add.items():
#     pooled_processes[0]['Washington'][key] = value

# pooled_processes[0]['Washington']

# set_of_entities = set()
# set_of_uris = []

# unified_dict = defaultdict(list)
# uris = set()
#
#
# for pooled_process in pooled_processes:
#     for target_entity, returned_queries in pooled_process.items():
#         for returned_query in returned_queries:
#             uri = returned_query['uri']
#             if uri not in uris:
#                 uris.add(uri)
#                 unified_dict[target_entity].append(returned_query)
#                 # unified_list.append(returned_query)
#
# print(len(unified_dict['Washington']))

# print(unified_list[:5])
# print(len(unified_list))
            # break

            # print(f'{target_entity}\t{returned_query}')
            # set_of_uris.append(returned_query['uri'])
        # print(target_entity)
        # print(returned_queries[0])
        # exit(1)
        # set_of_entities.add(target_entity)
        # for returned_query in returned_queries:
        #     uri = returned_query['uri']
        #     if not any(uri == query['uri'] for query in set_of_uris):
        #         set_of_uris.append(returned_query)
        #     else:
        #         print(returned_query)

# print(len(set_of_uris))
# # print(set_of_uris)
# print(len(set(set_of_uris)))

#
# print()
# print(set_of_entities)
# print(len(set_of_uris))

# list_of_uris = [dict(t) for t in {tuple(d.items()) for d in pooled_processes}]
# print(list_of_uris)

# for pooled_process in test_list:
#     for k, v in d.items():
#         print(k)
#         print(len(v))
#         exit(1)
