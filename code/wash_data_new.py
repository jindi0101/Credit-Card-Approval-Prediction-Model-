import pandas as pd


# record the worst credit record for single ID
credit_status = pd.read_csv("./wash_data/credit_record.csv")
credit_status.STATUS.replace('X', 0, inplace=True)
credit_status.STATUS.replace('C', 0, inplace=True)
credit_status.STATUS = credit_status.STATUS.astype('int')
credit_record = credit_status.groupby('ID').STATUS.max()

# merge two csv
application = pd.read_csv("./wash_data/application_record.csv")
df = pd.merge(application, credit_record, how='inner', on=['ID'])
df = df.sort_values('STATUS',ascending=False)

#drop duplicates for IDs with same features
df = df.drop_duplicates(subset=['CODE_GENDER',
                                'DAYS_BIRTH',
                                'DAYS_EMPLOYED',
                                'AMT_INCOME_TOTAL',
                                'NAME_FAMILY_STATUS',
                                'OCCUPATION_TYPE'
                                ],
                        keep='first', inplace=False)

#Change status to reject
df.STATUS = df.STATUS.apply(lambda x: 1 if x >= 2 else 0)
df = df.sort_values('AMT_INCOME_TOTAL')
df = df.reset_index(drop=True)
df.ID=df.index
df.columns = ('User_id',
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
              )
df.to_csv('./wash_data/data_pd.csv', index=0)
