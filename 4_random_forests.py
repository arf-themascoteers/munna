from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pandas as pd
from sklearn import model_selection
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor

df = pd.read_csv("Housing.csv")
array = df.to_numpy()

train, test = model_selection.train_test_split(array, test_size=0.2)

train_X = train[:,0:-1]
train_y = train[:,-1]
test_X = test[:,0:-1]
test_y = test[:,-1]


model = RandomForestRegressor()
model.fit(train_X, train_y)
predicted_y = model.predict(test_X)
score = r2_score(test_y, predicted_y)
print(score)

# Normalize: from sklearn.preprocessing import MinMaxScaler
