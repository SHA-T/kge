"""
This script filters out rows from the test or valid dataset that contain drugs, that are not seen in train set. 
This script is not needed if the dataset splits were made properly. 
However, this script is needed for the special datasets where we TRAIN on a completely different dataset 
(like external dataset, e.g.: drugs-indications) and TEST on dataset of interest (e.g. Yamanishi).
"""


import pandas as pd
import numpy as np


DATA_PATH = 'data_k_fold/train_on_external_valtest_on_yamanishi/ion_channel/with_side_effects/1'
MOVE_HALF_OF_TEST_TO_VALID = True           # Do this if you are using whole yamanishi subset as test


def split_dataframe(df):
    split_index = len(df) // 2
    df1, df2 = np.split(df, [split_index])
    return df1, df2


if __name__ == '__main__':
    df_test = pd.read_csv(f"{DATA_PATH}/test.txt", sep='\t',
                          names=['head', 'relation', 'tail'])
    df_train = pd.read_csv(f"{DATA_PATH}/train.txt", sep='\t',
                           names=['head', 'relation', 'tail'])
    
    # Get the unique values from column 'head' of dataframe df_train
    unique_values = set(df_train['head'])

    # Filter rows in dataframe df_test based on values in column 'tail'
    df_test_filtered = df_test[df_test['tail'].isin(unique_values)]

    print(df_test)
    print(df_test_filtered)

    if MOVE_HALF_OF_TEST_TO_VALID:
        df_test_filtered, df_valid_filtered = split_dataframe(df_test_filtered)
        print(df_test_filtered)
        print(df_valid_filtered)
        df_valid_filtered.to_csv(f"{DATA_PATH}/valid.txt", sep="\t", index=False, header=False)
        
    df_test_filtered.to_csv(f"{DATA_PATH}/test.txt", sep="\t", index=False, header=False)
    