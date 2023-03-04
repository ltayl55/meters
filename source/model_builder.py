
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd

from source.data_preprocess import DataPreprocessing



class ModelBuilder(DataPreprocessing):
    def __init__(self, *args, **kwargs):
        super(ModelBuilder, self).__init__(*args, **kwargs)

    def dt(self, X_train, X_test, y_train, y_test):
        #Create DT model
        DT_classifier = DecisionTreeClassifier()

        #Train the model
        DT_classifier.fit(X_train, y_train)

        #Test the model
        DT_predicted = DT_classifier.predict(X_test)

        error = 0
        for i in range(len(y_test)):
            error += np.sum(DT_predicted != y_test)

        total_accuracy = 1 - error / len(y_test)

        #get performance
        self.accuracy = accuracy_score(y_test, DT_predicted)

        return DT_classifier
    
    import pandas as pd

import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class PreProcess():
    def __init__(self):
        pass

    def load_data(self, path):
        data = pd.read_csv(path, sep="\t", header=None).dropna()

        print(data.head())

        data = data.to_numpy()

        return data

class ANN(PreProcess):
    def __init__(self, *args, **kwargs):
        super(ANN, self).__init__(*args, **kwargs)

    def myANN(self, myX_train,  myX_test, myy_train, myy_test):

        mlp = MLPClassifier(hidden_layer_sizes=(1, 3), learning_rate_init=0.09, max_iter=2500)

        mlp.fit(myX_train, myy_train)

        mlp_predicted = mlp.predict(myX_test)

        myerror = 0
        for i in range(len(myy_test)):
            myerror += np.sum(mlp_predicted != myy_test)

        mytotal_accuracy = 1 - myerror / len(myy_test)

        self.accuracy = accuracy_score(myy_test, mlp_predicted)

        return mlp