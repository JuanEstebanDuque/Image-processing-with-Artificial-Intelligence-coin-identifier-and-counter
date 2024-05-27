'''
Adaptación del código entregado en clase por el profesor
Milton Orlando Sarria
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from glob import glob 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

## leer todos los archivos
class coin_classifier:
    def __init__(self):
        self.root = 'Monedas'
        self.image_size = (100,100)
        self.test_size = 0.3
        self.max_iter = 1500
        self.model = None
        self.pca = None
        self.clases = None
        self.dic_clases = None
        self.names = None

    def loadData(self):

        file_names = glob(self.root+"/**/*.jpg", recursive=True)
        clases = glob(self.root+"/*")
        clases = clases=np.array([clase.split('\\')[1] for clase in clases])
        print(clases)

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
        self.dic_clases = dic_clases
        print('Done loading')
        return X, names


    def train(self, X, names):
        x_train, x_test, y_train, y_test = train_test_split(X,names,test_size = 0.3)

        pca = PCA() #n_components=0.99
        pca.fit(x_train)

        x_train=pca.transform(x_train)
        x_test=pca.transform(x_test)

        self.pca = pca

        print(f"[INFO] train set: {x_train.shape}, test set: {x_test.shape}")
        print("[INFO] entrenando LogisticRegression....")

        #entrenar un Regresión logistica
        clf = LogisticRegression(solver='lbfgs', max_iter=self.max_iter)
        clf.fit(x_train,y_train)
        self.model = clf

        print("[INFO] evaluando LogisticRegression....")

        y = clf.predict(x_test)
        acc =(y==y_test).sum()/y.size*100
        print('[INFO] Porcentaje de prediccion : ',acc)

    def Load(self):
        X, names = self.loadData()
        self.train(X,names)

    def predict(self, X):
        if self.model is None or self.pca is None:
            raise Exception("El modelo todavia no esta entrenado")
        
        frame = cv2.resize(X, (100, 100))  
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.array(gray).ravel()
        frames = []
        frames.append(gray)
        XData=np.vstack(frames)/255
        
        X_transformed = self.pca.transform(XData)
        
        result = (self.model.predict(X_transformed))[0]
        
        name = self.dic_clases[result]
        
        key_base = "_".join(name.split("_")[:1])
        numb = int(key_base)

        return numb
    