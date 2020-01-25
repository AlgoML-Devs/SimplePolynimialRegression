 
#Imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class Model:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def setDataFrame(self):
        df = pd.read_csv("bitcoin_cash_price.csv")
        
        return df
  
    
    def func(self):
      


       

        model = np.poly1d(np.polyfit(self.x, self.y, 3))
        line = np.linspace(1, 22, 100)

        ax = plt.subplot()
        plt.scatter(self.x, self.y)
        ax.plot(line, model(line), c="orange", label='Prices')
        print(model(5))
        ax.legend()
        
        plt.show()
    
    def linealRegression(self):
    
        x_train = self.x[:80]
        y_train = self.y[:80]
        

        x_test = self.x[80:]
        y_test = self.y[80:]

        lm = LinearRegression()
        lm.fit(x_train, y_train)
        predictions = lm.predict(x_test)
        print(predictions)
        plt.scatter(x_train, y_train, c="blue")
        plt.plot(x_test, predictions)

        
    def funcTest(self):
        col_op = self.DataFrame['Open']

    
        

x_values = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
y_values = [10,20,30,40,50,66,72,83,92,110,52,20,130,140,150,160,170]

obj = Model(x_values, y_values)
obj.funcTest()

