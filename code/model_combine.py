import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import tuo
import numpy as np


class Model550:
    '''
    Get the data from firstly cleaned csv file.
    Clean the data with more steps: normalize every data,
                                    transfer type columns to one-hot encoding,
                                    generate train set and test set.
    Use five method to generate model: neural network,
                                       xgboost,
                                       lightgbm,
                                       random forest,
                                       support vector machine.
    Generate a pandas dataframe to store the record.
    '''
    def __init__(self, data_file='old', smote=True):
        self.model_name = ''
        self.data_file = data_file
        self.data = self.input_data()
        self.smote = smote
        self.features, self.labels = self.data_cleaning()
        self.model_record = pd.DataFrame(np.array([[0, 0, 0, 0],
                                                   [0, 0, 0, 0],
                                                   [0, 0, 0, 0],
                                                   [0, 0, 0, 0],
                                                   [0, 0, 0, 0]
                                                   ]),
                                         columns=['Accuracy',
                                                  'Precision',
                                                  'Recall',
                                                  'F1-score'],
                                         index=['Neural_Network',
                                                'Random_Forest',
                                                'Xgboost',
                                                'Support_Vector_Machine',
                                                'Light gbm'])
        self.X_train, self.X_test, self.y_train, self.y_test=train_test_split(self.features, self.labels,
                                                                                stratify=self.labels,
                                                                                test_size=0.3,
                                                                                random_state=123)

    def input_data(self):
        if self.data_file == 'new':
            data = pd.read_csv('./wash_data/data_pd.csv')
            return data
        else:
            data = pd.read_csv('./wash_data/data.csv')
            return data

    def data_cleaning(self):
        self.data['age_month'] = self.data.Days_birth.apply(lambda x: int(-x // 30.417))
        self.data['employed_month'] = self.data.Days_employed.apply(lambda x: int(-x // 30.417) if x < 0 else -1)
        data = self.data.drop(['Days_birth', 'Days_employed', 'Mobil'], axis=1)
        data.loc[data.Occupation_type == '', "Occupation_type"] = "NA"
        data = pd.get_dummies(data,
                              columns=['Gender', 'Car', 'Realty', "income_type", 'education_type', 'Family_status',
                                       'Housing_type', 'Occupation_type'])
        data = (data - data.min()) / (data.max() - data.min())

        X = data.drop(['Reject', 'User_id'], axis=1)
        y = data['Reject']
        if self.smote:
            X, y = SMOTE().fit_sample(X, y)
        return X, y

    def shuffle_train_set(self, rate=0.3):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.labels,
                                                                                stratify=self.labels,
                                                                                test_size=rate,
                                                                                random_state=123)

    def neural_network(self, layer=(128, 64, 64, 32, 32), epochs=100):
        from keras import models
        from keras import layers
        from keras import optimizers
        model = models.Sequential()
        model.add(layers.Dense(layer[0], activation='relu', input_shape=(self.X_train.shape[1],)))
        for dense in layer[1:]:
            model.add(layers.Dense(dense, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.summary()
        model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        history = model.fit(self.X_train, self.y_train,
                            epochs=epochs,
                            batch_size=128,
                            validation_data=(self.X_test, self.y_test)
                            )
        y_predict = model.predict(self.X_test)
        y_predict = tuo.round_result(y_predict)
        result = tuo.return_result(self.y_test, y_predict)
        self.record_model(info=result, model_name='Neural_Network')

    def xgboost(self):
        from xgboost import XGBClassifier
        xgb = XGBClassifier()
        xgbfit = xgb.fit(self.X_train, self.y_train)
        y_predict = xgbfit.predict(self.X_test)
        result = tuo.return_result(self.y_test, y_predict)
        self.record_model(info=result, model_name='Xgboost')

    def random_forest(self, n_estimators=50):
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=n_estimators)
        rffit = rf.fit(self.X_train, self.y_train)
        y_predict = rffit.predict(self.X_test)
        result = tuo.return_result(self.y_test, y_predict)
        self.record_model(info=result, model_name='Random_Forest')

    def lightgbm_model(self):
        import lightgbm as lgb
        lgb_train = lgb.Dataset(self.X_train, self.y_train)
        lgb_eval = lgb.Dataset(self.X_test, self.y_test, reference=lgb_train)
        gbm = lgb.train({"objective": "binary", "learing_rate": 0.05},
                        lgb_train,
                        valid_sets=lgb_eval,
                        verbose_eval=100,
                        num_boost_round=3000,
                        early_stopping_rounds=500
                        )
        y_pred = gbm.predict(self.X_test, num_iteration=gbm.best_iteration)
        result = tuo.return_result(self.y_test, np.rint(y_pred))
        self.record_model(info=result, model_name='Light gbm')

    def support_vector_machine(self):
        from sklearn import svm
        model = svm.SVC(kernel='linear', C=1)
        svmc = model.fit(self.X_train, self.y_train)
        y_predict = svmc.predict(self.X_test)
        result = tuo.return_result(self.y_test, y_predict)
        self.record_model(info=result, model_name='Support_Vector_Machine')

    def record_model(self, info=None, model_name=None):
        self.model_record.loc[model_name] = info

    def all_model(self):
        self.xgboost()
        self.random_forest()
        self.support_vector_machine()
        self.neural_network()
        self.lightgbm_model()
        print(self.model_record)


if __name__ == '__main__':
    model = Model550()
    model.all_model()
    print(model.model_record)
