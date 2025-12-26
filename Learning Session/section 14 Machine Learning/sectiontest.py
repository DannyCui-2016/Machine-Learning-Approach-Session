# from sklearn.family import Model
from sklearn.linear_model import LinearRegression
import numpy as np
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split 

X, y = np.arange(10).reshape((5,2)),range(5)
X
list(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train
y_train
X_test
y_test
model = LinearRegression()

model.fit(X_train,y_train)

predictions = model.predict(X_test)

# test