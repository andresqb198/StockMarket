import  pandas as pd
import numpy as np

s_and_p_500 = pd.read_csv('s_and_p_500',parse_dates=True,index_col = 0)
print(s_and_p_500.head(5))