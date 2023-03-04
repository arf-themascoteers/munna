from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
from sklearn import model_selection

df = pd.read_csv("Housing_Min.csv")
array = df.to_numpy()

train, test = model_selection.train_test_split(array, test_size=0.2)

train_X = train[:,0:-1]
train_y = train[:,-1]
test_X = test[:,0:-1]
test_y = test[:,-1]


model = LinearRegression()
model.fit(train_X, train_y)
predicted_y = model.predict(test_X)
print(predicted_y)
score = r2_score(test_y, predicted_y)
print(score)
print(model.coef_)
print(model.intercept_)
# y = mx + c
# price = model.coef_ * area + model.intercept_