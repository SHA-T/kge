import pandas as pd

"""
This script creates an entities.dict file from the train, valid and test set under the specified data sub-folder.
"""

DATA_PATH = 'data_k_fold/yamanishi/enzyme/with_indications_similarity0.05pct'
FOLD = [1, 2, 3, 4, 5]

if __name__ == '__main__':

    for f in FOLD:
        df_train = pd.read_csv(f"{DATA_PATH}/{f}/train.txt", sep='\t',
                            names=['head', 'relation', 'tail'])
        df_valid = pd.read_csv(f"{DATA_PATH}/{f}/valid.txt", sep='\t',
                            names=['head', 'relation', 'tail'])
        df_test = pd.read_csv(f"{DATA_PATH}/{f}/test.txt", sep='\t',
                            names=['head', 'relation', 'tail'])

        df_all = pd.concat([df_train, df_valid, df_test])
        df_entities = pd.concat([df_all['head'].drop_duplicates(), df_all['tail'].drop_duplicates()],
                                ignore_index=True).drop_duplicates()

        df_entities = df_entities.reset_index(drop=True)

        df_entities.to_csv(f"{DATA_PATH}/{f}/entities_NEW.dict", sep="\t", index=True,
                        header=False)
    