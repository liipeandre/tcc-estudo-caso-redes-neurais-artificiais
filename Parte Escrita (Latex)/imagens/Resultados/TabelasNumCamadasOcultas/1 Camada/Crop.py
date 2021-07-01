import glob, os
import cv2
import numpy as np

for file in glob.glob("*.png"):
    
    # Carrega a imagem
    img = cv2.imread(file)

    # Recorta uma regiao da imagem
    crop = img[95:340, 125:1555]

    # Exibe a imagem transformada
    cv2.imwrite(file, crop)

'''
# Carrega a imagem
img = cv2.imread("Resultado['X'].png")

# Recorta uma regiao da imagem
crop = img[95:340, 125:1550]

# Exibe a imagem transformada
cv2.imwrite("saida.png", crop)
'''