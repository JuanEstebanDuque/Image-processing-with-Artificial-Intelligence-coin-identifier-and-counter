import cv2
import numpy as np
import os
from Logistic import coin_classifier

coinClassifier = coin_classifier()
coinClassifier.Load()
cap = cv2.VideoCapture(0)
cap.set(3, 4000)  # Ajustar resolución (opcional)
destino_circulos = 'Monedas/'
#fotos_por_moneda = int(input("Numero de fotos por moneda: "))


while True:
    k=0
        
    while True:
        total_de_dinero = []
        ret, frame = cap.read()
        output = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Procesamiento para detección de círculos
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        gray_blurred = cv2.medianBlur(gray_blurred, 5)
        gray = cv2.adaptiveThreshold(gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3.5)
        kernel = np.ones((5, 5), np.uint8)
        gray = cv2.erode(gray, kernel, iterations=1)
        gray = cv2.dilate(gray, kernel, iterations=1)

        detected_circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=50, param1=50, param2=30,
                                            minRadius=30, maxRadius=100)

        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles[0, :]))
            for (x, y, r) in detected_circles:
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                cv2.circle(output, (x, y), 1, (0, 0, 255), 3)
                # Recortar el área del círculo y guardar la imagen
                circle_area = frame[y - r:y + r, x - r:x + r]
                
                #IMPORTANTE
                #enviar al clasificador aqui (redimencionar la imagen) Se puede usar el de recortar la cara de las imagenes
                try:
                    total_de_dinero.append(coinClassifier.predict(circle_area))
                except cv2.error as e:
                    print("Error al procesar la imagen")
            

        cv2.imshow('Detección de círculos', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print("Total de dinero: ", sum(total_de_dinero))
        
    cv2.destroyAllWindows()
    cont = input("Desea registrar otro cliente? (S/N): ")
    if cont.upper()=='N':
        continuar = False

# Liberar la captura y cerrar la ventana
    cap.release()
    cv2.destroyAllWindows()
