#pip install pandas
#pip install matplotlib
#pip install scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Generación de dataframe a partir del CSV
data = pd.read_csv('Crop_recommendation.csv')

# Visualización de las primeras 5 filas para conocer la estructura de los datos
#print(data.head())

# Obtener información general sobre el DataFrame, incluyendo tipos de datos y conteo de valores no nulos
#print(data.info())

# Veamos qué valores asume labels
labels = data['label'].unique()
# print(labels)

# Vamos a codificar esta columna
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])
# Veamos el resultado
labels = data['label'].unique()
#print(labels)

# Resumen estadístico de los datos numéricos
#print(data.describe())


# Visualizar distribuciones de variables numéricas
#sns.pairplot(data)
#plt.show()


# Identificar valores atípicos (outliers) en las variables numéricas
sns.boxplot(data=data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']])
#plt.show()

# Eliminación de outliers
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
# Calcula el rango intercuartílico (IQR)
IQR = Q3 - Q1
# Define los límites superior e inferior para identificar outliers
lower_bound = Q1 - 0.3 * IQR
upper_bound = Q3 + 0.3 * IQR
# Filtra el DataFrame para eliminar outliers
print("Longitud del dataframe antes de filtrar: "+str(data.shape[0]))
data = data[((data < lower_bound) | (data > upper_bound)).any(axis=1)]
print("Longitud del dataframe después de filtrar: "+str(data.shape[0]))


# Chequeamos en el gráfico
sns.boxplot(data=data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']])
plt.show()

# Vemos que los datos tienen mucha varianza, así que sacar muchos outliers significa quedarse
# con muy pocos registros. Usamos un factor de 0.3 para quedarnos con una cantidad
# significativa de registros.

# Aplicamos Min-Max scaling para llevar las características a un rango común
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']])
data_scaled_minmax = pd.DataFrame(scaled_data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
#print(data_scaled_minmax.describe())


# También hacemos Z-score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']])
data_scaled_z = pd.DataFrame(scaled_data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
#print(data_scaled_z.describe())