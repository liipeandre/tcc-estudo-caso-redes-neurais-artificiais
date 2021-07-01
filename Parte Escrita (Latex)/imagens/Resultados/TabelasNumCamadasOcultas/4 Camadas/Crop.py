import glob, os
import cv2
import numpy as np

for file in glob.glob("*.png"):
    
    # Carrega a imagem
    img = cv2.imread(file)

    # Recorta uma regiao da imagem
    crop = img[95:340, 453:1295]

    # Exibe a imagem transformada
    cv2.imwrite(file, crop)

'''
# Carrega a imagem
img = cv2.imread("Resultado[10, 'X'].png")

# Recorta uma regiao da imagem
crop = img[95:340, 453:1295]

# Exibe a imagem transformada
cv2.imwrite("saida.png", crop)
'''