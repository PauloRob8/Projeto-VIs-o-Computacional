import cv2
import numpy as np

image = cv2.imread("dados.jpeg")

#Tons de Cinza
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#Filtro Blur
blurImg = cv2.GaussianBlur(gray_image,(15,15),0)
#Binarização
ret,tresh = cv2.threshold(blurImg,127,255,cv2.THRESH_BINARY)

#Detecção de bordas através do método Canny
bordas = cv2.Canny(tresh,70,150)

#Contando contornos externos cm RETR_EXTERNAL
(lixo,objetos,lixo) = cv2.findContours(bordas.copy(),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow("Dados sem filtro", image)
cv2.imshow("Dados - cinza", gray_image)
cv2.imshow("Dados - blur",blurImg)
cv2.imshow("Dados - binario",tresh)
cv2.imshow("Detectando bordas",bordas)
image2 = image.copy()

#Desenhando contorno nos objetos detectados
cv2.drawContours(image2,objetos,-1,(255,0,0),2)
fonte = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image2,'Quantidade de objetos detectados:' +str(len(objetos)),
            (10,20),fonte,0.5,0,0,cv2.LINE_AA)
cv2.imshow("Resultado",image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
