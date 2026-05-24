import pandas as pd
import matplotlib as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# import dataset
data=pd.read_csv("dataset.csv")
# data=pd.DataFrame(data)

#data spliting on the based of feature and target variable
X=pd.DataFrame(data["area"])
Y=pd.DataFrame(data["price"])

#prepared data for training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# create model
regressor = LinearRegression()


# traning model
regressor.fit(x_train,y_train)

# testing 
y_pred = (regressor.predict(x_test))
