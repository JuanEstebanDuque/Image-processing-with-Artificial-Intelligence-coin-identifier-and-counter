'''
Milton Orlando Sarria
Ejemplo para entrenar un sistema base empleando regresion logistica
se hace uso de las imagenes que han sido recolectadas previamente
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from glob import glob 
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

## leer todos los archivos
root = 'Monedas'
file_names = glob(root+"/**/*.jpg", recursive=True)
clases = glob(root+"/*")
clases = clases=np.array([clase.split('\\')[1] for clase in clases])

print(clases)
#leer las imagenes
frames = []
labels = []
dic_clases = {}
for file_name in tqdm(file_names):  
    
    frame = cv2.imread(file_name)
    frame = cv2.resize(frame, (100, 100))  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.array(gray).ravel()
    clase = file_name.split('\\')[1]

    label = np.where(clases==clase)[0][0]
    labels.append(label)
    frames.append(gray)
    dic_clases[label]=clase
#convertir a array
names = np.array(labels)
X=np.vstack(frames)/255
print('Done loading') 

image = X[0].reshape((100,100))
print(names[0])
plt.imshow(image,cmap="gray");
plt.show()

  
ACC = [] 
x_train, x_test, y_train, y_test = train_test_split(X,names,test_size = 0.3)

#normalizar
pca = PCA() #n_components=0.99
pca.fit(x_train)

x_train=pca.transform(x_train)
x_test=pca.transform(x_test)


print(f"[INFO] train set: {x_train.shape}, test set: {x_test.shape}")
print("[INFO] entrenando....")
#entrenar un clasificador simple
clf = LogisticRegression(solver='lbfgs', max_iter=1000)
clf.fit(x_train,y_train)
print("[INFO] evaluando....")
y = clf.predict(x_test)
acc =(y==y_test).sum()/y.size*100
print('[INFO] Porcentaje de prediccion : ',acc)

conf_matrix = confusion_matrix(y_test, y)

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
report = classification_report(y_test, y, target_names=clases)

# Imprimir el reporte de clasificación
print(report)