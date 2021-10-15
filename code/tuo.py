##############################
#Time: 10-25-2020
#Author: Tuo Sun
#There are some useful functions
################################
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def featureType(df):
    import numpy as np
    from pandas.api.types import is_numeric_dtype

    columns = df.columns
    rows = len(df)
    colTypeBase = []
    colType = []
    for col in columns:
        try:
            try:
                uniq = len(np.unique(df[col]))
            except:
                uniq = len(df.groupby(col)[col].count())
            if rows > 10:
                if is_numeric_dtype(df[col]):

                    if uniq == 1:
                        colType.append('Unary')
                        colTypeBase.append('Unary')
                    elif uniq == 2:
                        colType.append('Binary')
                        colTypeBase.append('Binary')
                    elif rows / uniq > 3 and uniq > 5:
                        colType.append('Continuous')
                        colTypeBase.append('Continuous')
                    else:
                        colType.append('Continuous-Ordinal')
                        colTypeBase.append('Ordinal')
                else:
                    if uniq == 1:
                        colType.append('Unary')
                        colTypeBase.append('Category-Unary')
                    elif uniq == 2:
                        colType.append('Binary')
                        colTypeBase.append('Category-Binary')
                    else:
                        colType.append('Categorical-Nominal')
                        colTypeBase.append('Nominal')
            else:
                if is_numeric_dtype(df[col]):
                    colType.append('Numeric')
                    colTypeBase.append('Numeric')
                else:
                    colType.append('Non-numeric')
                    colTypeBase.append('Non-numeric')
        except:
            colType.append('Issue')

    # Create dataframe
    df_out = pd.DataFrame({'Feature': columns,
                           'BaseFeatureType': colTypeBase,
                           'AnalysisFeatureType': colType})
    return df_out


def missing_values_table(df):
    # source: https://stackoverflow.com/questions/26266362/how-to-count-the-nan-values-in-a-column-in-pandas-dataframe
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    return mis_val_table_ren_columns


def round_result(pre_array):
    output_binary = []
    for result in pre_array[:]:
        if type(result) == np.ndarray or type(result) == list:
            output_binary.append(round(result[0]))
        else:
            output_binary.append(round(result))
    return output_binary


def print_result(y1, y2):
    y_true = list(map(int, list(y1)))
    y_pred = list(map(int, list(y2)))
    Accuracy = accuracy_score(y_true, y_pred)  # 注意没有average参数
    Precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1score = f1_score(y_true, y_pred, average='binary')
    print(Accuracy, Precision, recall, f1score)

def return_result(y1, y2):
    y_true = list(map(int, list(y1)))
    y_pred = list(map(int, list(y2)))
    Accuracy = accuracy_score(y_true, y_pred)  # 注意没有average参数
    Precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1score = f1_score(y_true, y_pred, average='binary')
    return Accuracy, Precision, recall, f1score