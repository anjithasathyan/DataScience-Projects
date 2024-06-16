import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

salary_data=pd.read_csv("Salary_Data.csv")
x = salary_data.drop('Salary',axis = 1)
y = salary_data.drop('YearsExperience',axis = 1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor=regressor.fit(x_train,y_train)

pickle.dump(regressor,open('model.pkl','wb'))
