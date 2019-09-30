import pandas as pd
import numpy as np
import math
from sklearn import preprocessing,svm
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime as datetime

style.use('ggplot')

s_and_p_500 = pd.read_csv('D:\ProyectoModelos\s_and_p_500.csv', parse_dates = True, index_col = 0)

plt.figure(1)
s_and_p_500['Close'].plot()
plt.legend(loc=4)
plt.xlabel('Fecha')
plt.ylabel('Precio')



df = s_and_p_500[['Open','High','Low','Close','Volume']]
df['High-Low percent']= (df['High']-df['Low'])/df['Close']*100
df['Percent change']=(df['Close']-df['Open'])/df['Open']*100
df=df[['Close','High-Low percent','Percent change','Volume']]

columna_pronostico = 'Close'

salida_pronostico = int(math.ceil(0.01*len(df)))
df['etiqueta'] = df[columna_pronostico].shift(-salida_pronostico)

X = np.array(df.drop(['etiqueta'],1))
X = preprocessing.scale(X)
X= X[:-salida_pronostico]
X_lately= X[-salida_pronostico:]

df.dropna(inplace=True)
y = np.array(df['etiqueta'])
y = np.array(df['etiqueta'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)
clf = LinearRegression()
clf.fit(X_train,y_train)
exactitud = clf.score(X_test,y_test)

conjunto_pronostico = clf.predict(X_lately)

print(conjunto_pronostico, exactitud,salida_pronostico)
df['Pronosticos'] = np.nan

ultima_fecha = df.iloc[-1].name
ultimo_unix = ultima_fecha.timestamp()
un_dia = 86400
siguiente_unix = ultimo_unix + un_dia

for i in conjunto_pronostico:
    siguiente_fecha = datetime.datetime.fromtimestamp(siguiente_unix)
    siguiente_unix += un_dia
    df.loc[siguiente_fecha] = [np.nan for _ in range(len(df.columns)-1)]+ [i]

plt.figure(2)
df['Close'].plot()
df['Pronosticos'].plot()
plt.legend(loc=4)
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.show()

















