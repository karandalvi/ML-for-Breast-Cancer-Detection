import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error
import numpy

source = pd.read_csv("breastCancerWisconsin.csv")

source = source[:].replace("?",numpy.NaN)
source.dropna(inplace=True)

Y = source["class"]

del source["id"]
del source["class"]
X = source

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=0)

gbc = ensemble.GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=3,
                                 max_features=1, min_samples_leaf=2)

gbc.fit(Xtrain, Ytrain)

joblib.dump(gbc, "breastCancerTrainedModel.pkl")

errorRate = mean_absolute_error(Ytrain, gbc.predict(Xtrain))
print("Training Data Error Rate:", errorRate)

errorRate = mean_absolute_error(Ytest, gbc.predict(Xtest))
print("Test Data Error Rate:", errorRate)
