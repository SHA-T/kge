"""
This script can be used to evaluate how useful the injected external data is.

It looks for entities p1 (which are connected to an external node e), that have missing links to neighbors
of another entity p2 (which is connected to the same external node e) in the train set, which (the missing links) are present in the test set.
Thus, these are missing links that can be found easily by a KGE model after the injection of the external data.

Same explanation put in different words:
The script looks for pairs of internal entities (p1, p2). p1 and p2 have to be connected to the same external entity e in the train set. 
In the test set p1 is connected to a subset of neighbors of p2 in the train set. These neighbors are all internal entities. They are proteins
if p1 and p2 are drugs. They are drugs if p1 and p2 are proteins.

Missing Links are saved as json files of the following form:
{
    p1: [
        neighbor1 of p1 in test and p2 in train,
        neighbor2 of p1 in test and p2 in train,
        ...
        ],
    ...
}

Finally, the script prints out a score that indicates how useful the external information is.
"""


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
import itertools


TARGET_PATH = "data_k_fold/yamanishi"
SUB_DATASET_NAMES = ['enzyme', 'gpcr', 'ion_channel', 'nuclear_receptor']       # enzyme | gpcr | ion_channel | nuclear_receptor | whole_yamanishi
NUMBER_OF_FOLDS = 5                                 # 1 | 2 | ... | k
EXTERNAL_DATA_TYPE = "indications"                  # indications | side_effects | ...
EXTERNAL_DATA_POSITION = "tail"                     # head | tail


def set_default(obj):
    """
    Returns a list version of obj, if obj is a set.
    """
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


if __name__ == '__main__':

    print("\nDictionary of entities p1 (which are connected to an external node e), that have missing links to neighbors")
    print("of another entity p2 (which is connected to the same external node e) in the train set, which (the missing links) are present in the test set.")
    print("Thus, these are missing links that can be found easily by a KGE model after the injection of the external data:\n")

    for sub_dataset in SUB_DATASET_NAMES:

        base_path = f"{TARGET_PATH}/{sub_dataset}"

        for fold in range(1, NUMBER_OF_FOLDS + 1):

            # DataFrame Representation
            df_train_ori = pd.read_csv(f"{base_path}/original/{fold}/train.txt",
                                       sep='\t', names=['head', 'relation', 'tail'])
            df_train_ext = pd.read_csv(f"{base_path}/with_{EXTERNAL_DATA_TYPE}/{fold}/train.txt",
                                       sep='\t', names=['head', 'relation', 'tail'])
            df_test = pd.read_csv(f"{base_path}/original/{fold}/test.txt",
                                  sep='\t', names=['head', 'relation', 'tail'])
            df_ext_only = pd.concat([df_train_ori, df_train_ext]).drop_duplicates(keep=False)

            # Graph Representation
            gr_train_ori = nx.from_pandas_edgelist(df_train_ori, 'head', 'tail')
            gr_test = nx.from_pandas_edgelist(df_test, 'head', 'tail')
            gr_ext_only = nx.from_pandas_edgelist(df_ext_only, 'head', 'tail')

            ml = {}

            # Set of external nodes (e.g. indications)
            ext_nodes = set()
            for index, row in df_ext_only.iterrows():
                ext_nodes.add(row[EXTERNAL_DATA_POSITION])

            for e in ext_nodes:
                # All neighbors of external node e --> list
                neighbors = [n for n in gr_ext_only.neighbors(e)]

                # All pair combinations of neighbors of e --> list of tuples
                pairs = list(itertools.combinations(neighbors, 2))

                # For pair in pairs
                for (p1, p2) in pairs:
                    # Direct neighbors of p1 & p2 in train & test dataset
                    neighbors_p1_train = set([n for n in gr_train_ori.neighbors(p1)])
                    neighbors_p2_train = set([n for n in gr_train_ori.neighbors(p2)])
                    try:
                        neighbors_p1_test = set([n for n in gr_test.neighbors(p1)])
                    except nx.exception.NetworkXError:
                        neighbors_p1_test = set()
                    try:
                        neighbors_p2_test = set([n for n in gr_test.neighbors(p2)])
                    except nx.exception.NetworkXError:
                        neighbors_p2_test = set()

                    # Missing links to neighbors of opposing pair px in train set, that are present in test set
                    ml_p1 = neighbors_p1_test.intersection(neighbors_p2_train).difference(neighbors_p1_train)
                    ml_p2 = neighbors_p2_test.intersection(neighbors_p1_train).difference(neighbors_p2_train)

                    # Add those missing links to the dictionary
                    if p1 in ml:
                        ml[p1].update(ml_p1)
                    else:
                        ml[p1] = ml_p1
                    if p2 in ml:
                        ml[p2].update(ml_p2)
                    else:
                        ml[p2] = ml_p2

            # Delete empty keys and prepare scoring
            ml = {k: v for k, v in ml.items() if v}
            ml_count = sum(len(v) for v in ml.values())
            ext_triples_count = df_ext_only.shape[0]

            # For print:
            # print(json.dumps(ml, indent=4, default=set_default))
            dump = json.dumps(ml, indent=4, default=set_default)
            f = open(f"ml_{sub_dataset}_fold_{fold}.json", "w+")
            f.write(dump)
            f.close()
            print(f"{sub_dataset}, Fold {fold}:")
            print("Number of missing links:", ml_count)
            print("Number of external triples:", ext_triples_count)
            print("score:", round((ml_count / ext_triples_count), 4), "\n")

            # plt.figure(figsize=(10, 8))
            # nx.draw(gr_test, with_labels=True)
            # plt.show()