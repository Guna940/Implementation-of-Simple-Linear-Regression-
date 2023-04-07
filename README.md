# Implementation of Simple Linear Regression Model for Predicting the Marks Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```txt
1. Use the standard libraries in python for Gradient Design.
2.Set Variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing the graph.
5.predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given data.
```

## Program:
```txt
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Gunanithi S
RegisterNumber:  212220220015
```
```python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print("df.head():")
df.head()
```
```python3
print("df.tail(): ")
df.tail()
```
```python3
print("Array values of x:")
x=df.iloc[:,:-1].values
x
```
```python3
print("Array value of y:")
y=df.iloc[:,1].values
y
```
```python3
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print("y_pred:")
y_pred
```
```python3
print("y_test:")
y_test
```
```python3
print("Training set graph:")
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours vs Scores (Trainig set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```python3
print("Test Set graph:")
plt.scatter(x_test,y_test,color="green")
plt.plot(x_test,regressor.predict(x_test),color="violet")
plt.title("Hours vs Scores (Trainig set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```python3
print("Values of MSE,MAE and RMSE:")
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
<img src="https://user-images.githubusercontent.com/89703145/230582188-c6a2f7bf-917d-4be7-ad89-8fceedf3d49f.png" alt="Screenshot" width="150">

<img src="https://user-images.githubusercontent.com/89703145/230582256-3efe098d-0677-40b4-8016-6957e64a7b98.png" alt="Screenshot" width="150">

<img src="https://user-images.githubusercontent.com/89703145/230582341-d5ff152e-ead0-4666-ad7e-3c396f4e14b1.png" alt="Screenshot" width="100">

<img src="https://user-images.githubusercontent.com/89703145/230582387-8f68b337-25ea-4a43-9f8b-393ac0aa8d2d.png" alt="Screenshot" width="350">

<img src="https://user-images.githubusercontent.com/89703145/230582524-4ef9aeac-7431-4836-a4c6-8540fec1d03d.png" alt="Screenshot" width="350">

<img src="https://user-images.githubusercontent.com/89703145/230582618-44caf05f-1fac-4162-ae76-6cfd4a2a5a18.png" alt="Screenshot" width="250">

<img src="https://user-images.githubusercontent.com/89703145/230582838-f84c1bea-41b6-4c54-88bf-bf878023aa65.png" alt="download" width="300">

<img src="https://user-images.githubusercontent.com/89703145/230583320-84b3a988-a906-42ed-93b1-4d71bbf35eb2.png" alt="download" width="300">

<img src="https://user-images.githubusercontent.com/89703145/230584088-76791c2a-b013-473d-bafe-026164e9f2de.png" alt="Screenshot" width="200">


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

