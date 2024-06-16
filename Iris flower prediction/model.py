import pandas as pd
import pickle

iris_data = pd.read_csv("iris .csv")
x = iris_data.drop('Classification',axis = 1)
y = iris_data['Classification']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.svm import SVC
sv = SVC(kernel='linear').fit(x_train,y_train)

pickle.dump(sv,open('model.pkl','wb'))
