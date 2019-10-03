from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('s_and_p_500.csv', parse_dates = True, index_col = 0)
df['High-Low percent']= (df['High']-df['Low'])/df['Close']*100
df['Percent change']=(df['Close']-df['Open'])/df['Open']*100

X = df[['Open']]
y = df['Close']
print(y.head())

plt.plot(X,y)
plt.show()
