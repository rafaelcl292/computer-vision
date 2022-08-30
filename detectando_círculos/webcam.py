#!/usr/bin/python
# -*- coding: utf-8 -*-

from dis import dis
import cv2
import os,sys, os.path
import numpy as np
import fotogrametria

# ->>> !!!! FECHE A JANELA COM A TECLA ESC !!!! <<<<-

def calcular_angulo_e_distancia_na_image_da_webcam(img, f):
    """Não mude ou renomeie esta função
        ->>> !!!! FECHE A JANELA COM A TECLA ESC !!!! <<<<-
        deve receber a imagem da camera e retornar uma imagems com os contornos desenhados e os valores da distancia e o angulo.
    """

    img2 = img.copy()


    try:
        h, centro_ciano, centro_magenta, img2 = fotogrametria.calcular_distancia_entre_circulos(img2)
        angulo = fotogrametria.calcular_angulo_com_horizontal_da_imagem(centro_ciano, centro_magenta)
        cv2.line(img2, centro_ciano, centro_magenta, (0, 255, 0), thickness=3, lineType=8)
        D = fotogrametria.encontrar_distancia(300, 14, h)
    except:
        D = 0
        angulo = 0
    
    return img2, D, angulo


def desenhar_na_image_da_webcam(img, distancia, angulo):
    """Não mude ou renomeie esta função
        ->>> !!!! FECHE A JANELA COM A TECLA ESC !!!! <<<<-
        deve receber a imagem da camera e retornar uma imagems com os contornos desenhados e a distancia e o angulo escrito em um canto da imagem.
    """
    
    cv2.putText(img, f'Angle = {angulo:.0f} degrees', (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
    cv2.putText(img, f'Distance = {distancia:.0f} cm', (10, 60), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
    return img


if __name__ == "__main__":
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    ## -> Mude o Foco <- ##
    f = fotogrametria.encontrar_foco(1, 1, 1)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        img, distancia, angulo = calcular_angulo_e_distancia_na_image_da_webcam(frame, f)
        img = desenhar_na_image_da_webcam(img, distancia, angulo)
        cv2.imshow("preview", img)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    cv2.destroyWindow("preview")
    vc.release()
