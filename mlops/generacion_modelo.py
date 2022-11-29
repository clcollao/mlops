import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn #
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, plot_roc_curve, confusion_matrix
from sklearn.model_selection import KFold

# Definición ruta acceso a los datos
DATA = Path('/Users/claudiocollaobahamondes/Desktop/mlops/mlops/data')

# Validación si existe la carpeta data
DATA.exists()

# Carga de los datos
df = pd.read_csv(DATA/'creditcard.csv')

#Dimensiones dataset original
print(df.shape)

# Creación de datasets con transacciones normales  y anormales
normal = df[df['Class']==0].sample(frac=0.5, random_state=123).reset_index(drop=True)
anormales = df[df['Class']==1]

# Dimensiones dataset normal y anormales
print(normal.shape)
print(anormales.shape)

# Generación de train, test y validación
normal_train, normal_test = train_test_split(normal, test_size=0.2, random_state=123)
anormales_train, anormales_test = train_test_split(anormales, test_size=0.2, random_state=123)
normal_train, normal_validate = train_test_split(normal_train, test_size=0.2, random_state=123)
anormales_train, anormales_validate = train_test_split(anormales_train, test_size=0.2, random_state=123)