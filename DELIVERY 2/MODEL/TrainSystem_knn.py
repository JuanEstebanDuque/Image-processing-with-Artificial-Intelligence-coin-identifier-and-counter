import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from glob import glob 
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

## Leer todos los archivos
root = 'Monedas'
file_names = glob(root+"/**/*.jpg", recursive=True)
clases = glob(root+"/*")
clases = np.array([clase.split('\\')[1] for clase in clases])

print(clases)
# Leer las imágenes
frames = []
labels = []
dic_clases = {}
for file_name in tqdm(file_names):  
    
    frame = cv2.imread(file_name)
    frame = cv2.resize(frame, (100, 100))  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.array(gray).ravel()
    clase = file_name.split('\\')[1]

    label = np.where(clases == clase)[0][0]
    labels.append(label)
    frames.append(gray)
    dic_clases[label] = clase

# Convertir a array
names = np.array(labels)
X = np.vstack(frames) / 255
print('Done loading') 

image = X[0].reshape((100, 100))
print(names[0])
plt.imshow(image, cmap="gray")
plt.show()

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(X, names, test_size=0.3)

# Entrenar el clasificador KNN
knn = KNeighborsClassifier(n_neighbors=17)  # Puedes ajustar el valor de k según tus necesidades
knn.fit(x_train, y_train)

# Evaluar el modelo
y_pred = knn.predict(x_test)
acc = (y_pred == y_test).sum() / y_test.size * 100
print('[INFO] Porcentaje de predicción:', acc)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Imprimir los resultados
print(conf_matrix.ravel())

# Calcular el reporte de clasificación
report = classification_report(y_test, y_pred, target_names=clases)

# Imprimir el reporte de clasificación
print(report)
