from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

train_X = [
    [500],
    [700],
    [1000],
    [1200],
    [1500]
]

train_y = [
    [1100],
    [1400],
    [1900],
    [2300],
    [2800]
]

test_X = [
    [600],
    [650],
    [1100],
    [1300],
    [1600]
]

test_y = [
    [1240],
    [1300],
    [2000],
    [2300],
    [3000]
]

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