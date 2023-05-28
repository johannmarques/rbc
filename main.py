import sidrapy
import pandas as pd
from matplotlib import pyplot
import seaborn as sns

# Obtaining data from SIDRA API
data = sidrapy.get_table(table_code= "1846",
                            territorial_level = "1",
                            ibge_territorial_code = "all",
                            period = "all",
                            classification = "11255/90687,90691,90696,90705,90706,90707,93404,93405,93406,93407,93408,102880")

# First row as header
data.columns = data.iloc[0]
data = data[1:]

# Drop some unecessary columns

data = data.drop(data.columns[[0, 1, 2, 3, 5, 6, 7, 9, 11, 12]], axis = 1)

# Quarterly date variable
data['date'] = pd.to_datetime(data[data.columns[1]].str.slice(-4)+
               "/"+
               data[data.columns[1]].str.slice(0,1).replace({"1" : "03",
                                                             "2" : "06",
                                                             "3" : "09",
                                                             "4" : "12"}),
               format = '%Y/%m')
data.index = data['date']
data['Valor'] = data['Valor'].astype('float64')

# First plot!
g = sns.FacetGrid(data, col="Setores e subsetores", height=2.5, col_wrap=3)
g.map_dataframe(sns.lineplot, x = 'date', y='Valor')

# Deseasonalizing

Y = data['Valor'].loc[data['Setores e subsetores'] == 'PIB a preços de mercado']
Quarter = data.loc[data['Setores e subsetores'] == 'PIB a preços de mercado'][data.columns[1]].str.slice(0,1)

import numpy as np
from sklearn.linear_model import LinearRegression

X = pd.get_dummies(Quarter, drop_first=False)
model_deseas = LinearRegression(fit_intercept = False)
model_deseas.fit(X = X, y = Y)

sns.lineplot(data = Y - model_deseas.predict(X) + np.mean(Y))