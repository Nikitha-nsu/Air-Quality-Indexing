# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 18:15:56 2020

@author: Nikitha
"""
from html_data import first_df
from aqi_data2 import avg_day

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#df_avg.head
print(df_scrap)

print(df_avg)
df_scrap= pd.concat([df_scrap,df_avg],axis=1)
df_scrap
df_scrap.to_csv('combine.csv')