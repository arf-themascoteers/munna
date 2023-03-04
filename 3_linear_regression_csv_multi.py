from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
from sklearn import model_selection

df = pd.read_csv("Housing.csv")
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
# y = a1x1 + a2x2 + a3x3 + a4x4 + a5x5 + b
# price =
#   model.coef_[0] * area
# + model.coef_[0] * bedrooms
# + model.coef_[0] * bathrooms
# + model.coef_[0] * stories
# + model.coef_[0] * parking
# + model.coef_[0] * price
# + model.intercept_

# Further: Categorical variable. One hot encoding.
# https://www.kaggle.com/datasets/yasserh/housing-prices-dataset/code?resource=download