##############################
# 9.31.2020
# Tuo Sun
#############################
import pandas as pd
import csv


def application_set():
    '''reform the basic data, generate mapping and new csv list

    This function will generate a mapping dictionary from unique user id to id, and get the
    information list for every user id.

    Arg(s):
        None
    Return(s):
        mapping(dict): map the user id to the transaction id(7 digits)
        new_table(list): a list of information.
    '''
    #load the data and sort the data by "AMT_INCOME_TOTAL", "DAYS_BIRTH". This will increase the efficiency
    application = pd.read_csv('./wash_data/application_record.csv')
    application.sort_values(by=["AMT_INCOME_TOTAL", "DAYS_BIRTH"], inplace=True)
    data = application.values.tolist()

    # read the csv line-by-line to group transaction ids with same information, in other words, the same person.
    mapping, old_info, new_id, new_table = {}, [], -1, {}
    for row in data:
        if row[1:] != old_info:
            new_id += 1
            new_table[new_id] = row[1:]
            mapping[new_id] = [row[0]]
            old_info = row[1:]
        else:
            mapping[new_id].append(row[0])
    return mapping, new_table


def credit_set():
    '''group the same transaction id, and record the worst record.

    This function will group the same transaction id(7 digits), and record the worst record.
    0 means this transaction was covered on time. 1 means this transaction is overdue for a long
    time, and the user in this id will be reject.

    Arg(s):
        None

    Return(s):
        new_table(dict): {transaction id: worst_transaction_record
    '''
    #load the data
    credit = pd.read_csv('./wash_data/credit_record.csv')
    data = credit.values.tolist()

    # more than 60 days over due is a discredit transaction
    credit_level = {'X': 0, 'C': 0, '0': 0, '1': 0, '2': 1, '3': 1, '4': 1, '5': 1}
    old_ID, new_table = 0, {}
    for row in data:
        if row[0] != old_ID:
            new_table[row[0]] = credit_level[row[2]]
            old_ID = row[0]
        else:
            new_table[row[0]] = max(new_table[row[0]], credit_level[row[2]])
    return new_table


def convert_sort_dict(old_dict):
    '''convert and sort a dictionary.

    This function will convert the key and the value. The value should be a list, and every element in the
    will become the new key. The key of the value will be convert to the value.
    This function generate a dictionary mapping id to the user id. And sort the output list by id.

    Arg(s):
        old_dict(dict): {key: value(list)

    Return(s)
        sorted_dict(dict): {value[0]: key, value[1]: key,}
    '''
    converted_dict = {}
    for key in old_dict:
        for ID in old_dict[key]:
            converted_dict[ID] = key

    # sort the keys first and then get the dictionary from value.
    sorted_dict = {}
    sorted_key = sorted(converted_dict)
    for key in sorted_key:
        sorted_dict[key] = converted_dict[key]

    return sorted_dict


def new_credit(mapping):
    '''get the worst transaction record for every user id.

    This function will connect user id to transaction id and record its worst transaction record
    for each user id.

    Args(s):
        mapping(dict): map user id to id

    Return(s):
        y_train(dict): {user_id: worst_transaction_id}
    '''
    credit_data = credit_set()
    sorted_map = convert_sort_dict(mapping)
    common_key = list(credit_data.keys() & sorted_map.keys())

    y_train = {}
    for key in common_key:
        if sorted_map[key] in y_train.keys():
            #only record the worst record
            if y_train[sorted_map[key]] == 1 or credit_data[key] == 1:
                y_train[sorted_map[key]] = 1
        else:
            y_train[sorted_map[key]] = credit_data[key]

    return y_train


if __name__ == '__main__':
    mapping, info_set = application_set()
    label = new_credit(mapping)
    csv_list = []

    #select the user id in both application record and credit record
    common_key = list(info_set.keys() & label.keys())
    common_key.sort()
    new_id = 0

    #combine the information table with the credit record
    for key in common_key:
        csv_list.append([new_id] + info_set[key] + [label[key]])
        new_id += 1

    with open("./wash_data/data.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['User_id',
                         'Gender',
                         'Car',
                         'Realty',
                         'children_count',
                         'income_amount',
                         'income_type',
                         'education_type',
                         'Family_status',
                         'Housing_type',
                         'Days_birth',
                         'Days_employed',
                         'Mobil',
                         'Work_phone',
                         'Phone',
                         'Email',
                         'Occupation_type',
                         'Count_family_members',
                         'Reject'
                         ])
        for row in csv_list:
            writer.writerow(row)

