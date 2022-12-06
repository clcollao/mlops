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
normal_train, normal_validate = train_test_split(normal_train, test_size=0.25, random_state=123)
anormales_train, anormales_validate = train_test_split(anormales_train, test_size=0.25, random_state=123)

print('Dimensiones Normal Train: ',normal_train.shape)
print('Dimensiones Normal Test: ',normal_test.shape)
print('Dimensiones Normal Validación: ',normal_validate.shape)
print()
print('Dimensiones Anormales Train: ',anormales_train.shape)
print('Dimensiones Anormales Test: ',anormales_test.shape)
print('Dimensiones Anormales Validación: ',anormales_validate.shape)

# Creación de  X Train, Test y Validation
x_train = pd.concat((normal_train,anormales_train))
x_test = pd.concat((normal_test,anormales_test))
x_validate = pd.concat((normal_validate,anormales_validate))

print('Dimensiones Train: ',x_train.shape)
print('Dimensiones Test: ',x_test.shape)
print('Dimensiones Validación: ',x_validate.shape)

# Creación de  y Train, Test y Validation
y_train = np.array(x_train['Class'])
y_test = np.array(x_test['Class'])
y_validate = np.array(x_validate['Class'])

# Eliminación columna Class en X
x_train = x_train.drop('Class',axis=1)
x_test = x_test.drop('Class',axis=1)
x_validate = x_validate.drop('Class',axis=1)

# Estandarizar los dataset
scaler = StandardScaler()
scaler.fit(pd.concat((normal,anormales)).drop('Class',axis=1))
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_validate = scaler.transform(x_validate)

# Entrenamiento del Modelo
sk_model = LogisticRegression(random_state=None,max_iter=400,solver='newton-cg').fit(x_train,y_train)


# Evaluación del Modelo
eval_acc = sk_model.score(x_test,y_test)

pred = sk_model.predict(x_test)
auc_score = roc_auc_score(y_test,pred)

print(f'AUC Score: {auc_score:.3%}')
print(f'Eval Accuracy: {eval_acc:.3%}')

# Curva ROC
roc_plot = plot_roc_curve(sk_model,x_test,y_test,name='Scikit-learn ROC Curve')
plt.show()

# Matriz de Confusión
conf_matrix = confusion_matrix(y_test,pred)
ax = sns.heatmap(conf_matrix,annot=True,fmt='g')
ax.invert_xaxis()
ax.invert_yaxis()
plt.ylabel('Actual')
plt.xlabel('Predicho')
plt.show()

# Validación del modelo
peso_anomalias = [1,5,10,15] # Lista de pesos a iterar
num_folds = 5  # Definir el número de fold
k_fold = KFold(n_splits=num_folds,shuffle=True,random_state=123) # Particiona los datos en diferentes folds

logs = []
for f in range(len(peso_anomalias)):
    fold = 1
    accuracies = []
    auc_scores = []
    for train, test in k_fold.split(x_validate,y_validate):
        peso = peso_anomalias[f]
        class_pesos = {
            0:1,
            1:peso
        }
        sk_model = LogisticRegression(
            random_state=None,
            max_iter=100,
            solver='newton-cg',
            class_weight=class_pesos).fit(x_validate[train],y_validate[train])
        for h in range(40): print('-',end='')
        print(f'\nfold {fold}\nPeso Anomalia:{peso}')

        eval_acc = sk_model.score(x_validate[test],y_validate[test])
        preds = sk_model.predict(x_validate[test])

        try:
            auc_score = roc_auc_score(y_validate[test],preds)
        except:
            auc_score = -1
    
        print('AUC: {}\neval_acc: {}'.format(auc_score,eval_acc))
        accuracies.append(eval_acc)
        auc_scores.append(auc_score)

        log = [sk_model, x_validate[test],y_validate[test],preds]
        logs.append(log)
        fold = fold + 1
    print('\nAverages: ')
    print('Accuracy: ',np.mean(accuracies))
    print('AUC: ',np.mean(auc_scores))
    print('')
    print('Best: ')
    print('Accuracy: ',np.max(accuracies))
    print('AUC: ',np.max(auc_scores))




