#### Análisis exploratorio ###############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Definición ruta acceso a los datos
DATA = Path('/Users/claudiocollaobahamondes/Desktop/mlops/mlops/data')

# Validación si existe la carpeta data
DATA.exists()

df = pd.read_csv(DATA/'creditcard.csv')

#### Análisis exploratorio ###############
print('El dataset tiene {} filas y {} columns'.format(df.shape[0],df.shape[1]))
print()
print(list(df.columns))
print()
print(df.info())

#### Análisis Variable Objetivo ##########
anomalias = df[df['Class']==1]
normal = df[df['Class']==0]

print('Dataframe Anomalias ',anomalias.shape)
print()
print('Dataframe Normal ',normal.shape)

sns.histplot(data=df,x='Class')
plt.show()





